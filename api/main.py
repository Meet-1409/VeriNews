"""
VeriNews FastAPI backend.

Endpoints
---------
GET  /api/health              → {"status": "ok", "models_ready": bool}
GET  /api/models              → list of available model info dicts
POST /api/analyze             → classify text, fetch articles, save to DB
GET  /api/result/{query_id}   → load a past result with articles
GET  /api/history             → paginated recent queries
POST /api/feedback            → save yes / no feedback for a query

Static frontend served from frontend/ at /.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT     = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="VeriNews API", version="2.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── lazy singletons ──────────────────────────────────────────────────────

_predictor = None
_db        = None


def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from src.inference.predictor import FakeNewsPredictor
            _predictor = FakeNewsPredictor()
            log.info("Predictor ready (default: ensemble)")
        except FileNotFoundError as exc:
            log.warning("Models not ready: %s", exc)
    return _predictor


def get_db():
    global _db
    if _db is None:
        from src.storage.database import VeriNewsDB
        _db = VeriNewsDB()
    return _db


# ── schemas ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text:             str  = Field(..., min_length=10, max_length=50_000)
    model_key:        str  = "ensemble"
    include_evidence: bool = False


class FeedbackRequest(BaseModel):
    query_id: int
    feedback: str = Field(..., pattern="^(yes|no)$")


# ── health ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "models_ready": get_predictor() is not None,
        "version":      "2.0.0",
    }


# ── models list ──────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    p = get_predictor()
    if p is None:
        return {"models": [], "error": "Models not trained yet. Run scripts/train_models.py"}
    return {"models": p.available_models()}


# ── analyze ──────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if len(text) < 10:
        raise HTTPException(400, "Text too short (min 10 characters).")

    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            503,
            "Models not ready. Please run:  python scripts/train_models.py",
        )

    # ── ML prediction ────────────────────────────────────────────────────
    ml = predictor.predict(text, model_key=req.model_key)

    # ── Persist query ────────────────────────────────────────────────────
    db       = get_db()
    query_id = db.save_query(
        query_text=text,
        label=ml["label"],
        confidence=ml["confidence"],
        prob_fake=ml["prob_fake"],
        prob_real=ml["prob_real"],
        model_key=ml["model_key"],
    )

    # ── Related articles (async, non-blocking) ───────────────────────────
    articles: list[dict] = []
    try:
        from src.models.model_manager import get_manager
        from src.retrieval.news_fetcher import fetch_verified_news
        from src.retrieval.similarity_ranker import rank_articles_by_similarity

        # Compact query: first 10 meaningful words
        words = [w for w in text.split() if len(w) >= 4][:10]
        query = " ".join(words)

        loop     = asyncio.get_event_loop()
        raw_arts = await loop.run_in_executor(None, lambda: fetch_verified_news(query))

        if raw_arts:
            mgr       = get_manager()
            vec       = mgr.get_tfidf()
            ranked    = rank_articles_by_similarity(text, raw_arts, vectorizer=vec)
            # Filter out bare homepage links (real articles have path depth ≥ 3)
            articles = [
                a for a in ranked
                if a.get("url", "").count("/") >= 4
            ][:5]

        if articles:
            db.save_verified_articles(query_id, articles)

    except Exception as exc:
        log.warning("Article fetch skipped: %s", exc)

    # ── Evidence brief (optional, requires API keys) ──────────────────────
    evidence: dict | None = None
    if req.include_evidence:
        try:
            from src.research.live_factcheck import LiveFactChecker
            lfc = LiveFactChecker()
            loop = asyncio.get_event_loop()
            evidence = await loop.run_in_executor(None, lambda: lfc.analyze(text, ml))
        except Exception as exc:
            log.warning("Evidence brief skipped: %s", exc)

    # ── Response ─────────────────────────────────────────────────────────
    explanation = ml.get("explanation", [])
    return {
        "query_id":    query_id,
        "label":       ml["label"],
        "prob_fake":   ml["prob_fake"],
        "prob_real":   ml["prob_real"],
        "confidence":  ml["confidence"],
        "model_key":   ml["model_key"],
        "threshold":   ml["threshold"],
        "explanation": [{"word": w, "weight": wt} for w, wt in explanation],
        "articles":    articles,
        "evidence":    evidence,
    }


# ── single result ────────────────────────────────────────────────────────

@app.get("/api/result/{query_id}")
def get_result(query_id: int):
    db  = get_db()
    row = db.get_query_by_id(query_id)
    if row is None:
        raise HTTPException(404, "Query not found.")
    articles = db.get_verified_articles_for_query(query_id)
    feedback = db.get_feedback_count(query_id=query_id)
    return {
        "query_id":   query_id,
        "text":       row["query_text"],
        "label":      row["label"],
        "confidence": row.get("confidence"),
        "prob_fake":  row.get("prob_fake"),
        "prob_real":  row.get("prob_real"),
        "model_key":  row.get("model_key", "ensemble"),
        "created_at": row.get("created_at"),
        "articles":   articles,
        "feedback":   feedback,
        # explanation not stored; front-end re-runs for a live explanation
        "explanation": [],
    }


# ── history ──────────────────────────────────────────────────────────────

@app.get("/api/history")
def history(limit: int = 50):
    db   = get_db()
    rows = db.get_recent_queries(limit=limit)
    result = []
    for row in rows:
        qid  = row["id"]
        arts = db.get_verified_articles_for_query(qid)
        fb   = db.get_feedback_count(query_id=qid)
        result.append(
            {
                "id":           qid,
                "text":         row["query_text"],
                "label":        row["label"],
                "confidence":   row.get("confidence"),
                "prob_fake":    row.get("prob_fake"),
                "prob_real":    row.get("prob_real"),
                "model_key":    row.get("model_key", "ensemble"),
                "created_at":   row.get("created_at"),
                "num_articles": len(arts),
                "feedback":     fb,
            }
        )
    return {"history": result}


# ── evidence brief (separate, no DB write) ────────────────────────────

class EvidenceRequest(BaseModel):
    text:      str = Field(..., min_length=10, max_length=50_000)
    ml_label:  str  = "Uncertain"
    ml_confidence: float = 0.5


@app.post("/api/evidence")
async def evidence_brief(req: EvidenceRequest):
    """Run the ChatGPT / Perplexity evidence brief without saving a new DB row."""
    try:
        from src.research.live_factcheck import LiveFactChecker
        lfc  = LiveFactChecker()
        loop = asyncio.get_event_loop()
        ml   = {"label": req.ml_label, "confidence": req.ml_confidence}
        ev   = await loop.run_in_executor(None, lambda: lfc.analyze(req.text, ml))
        return ev
    except Exception as exc:
        log.warning("Evidence brief error: %s", exc)
        raise HTTPException(503, f"Evidence brief failed: {exc}")


# ── feedback ─────────────────────────────────────────────────────────────

@app.post("/api/feedback")
def save_feedback(req: FeedbackRequest):
    db = get_db()
    db.save_feedback(req.query_id, req.feedback)
    return {"status": "saved"}


# ── static frontend ───────────────────────────────────────────────────────

if FRONTEND.exists():
    static_dir = FRONTEND / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index_page():
        return FileResponse(str(FRONTEND / "index.html"))

    @app.get("/result")
    def result_page():
        return FileResponse(str(FRONTEND / "result.html"))

    @app.get("/history")
    def history_page():
        return FileResponse(str(FRONTEND / "history.html"))
