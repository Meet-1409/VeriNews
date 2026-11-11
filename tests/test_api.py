"""
Integration tests for the FastAPI backend.

Run with:
    pytest tests/test_api.py -v

Uses FastAPI's TestClient (httpx under the hood). The predictor and DB are
mocked so no trained models or real DB are required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── shared mocks ──────────────────────────────────────────────────────────────

def _mock_predictor():
    p = MagicMock()
    p.predict.return_value = {
        "label":       "Real",
        "prob_fake":   0.18,
        "prob_real":   0.82,
        "confidence":  0.82,
        "model_key":   "ensemble",
        "threshold":   0.50,
        "explanation": [("scientists", 0.42), ("discover", 0.31)],
    }
    p.available_models.return_value = [
        {"key": "ensemble", "display_name": "Ensemble", "description": "...", "available": True, "metrics": {}},
        {"key": "lr",       "display_name": "Logistic Regression", "description": "...", "available": True, "metrics": {}},
    ]
    return p


def _mock_db():
    db = MagicMock()
    db.save_query.return_value = 42
    db.get_query_by_id.return_value = {
        "query_text": "Some news article",
        "label":      "Real",
        "confidence": 0.82,
        "prob_fake":  0.18,
        "prob_real":  0.82,
        "model_key":  "ensemble",
        "created_at": "2025-06-01T12:00:00",
    }
    db.get_verified_articles_for_query.return_value = []
    db.get_feedback_count.return_value = {"yes": 0, "no": 0}
    db.get_recent_queries.return_value = []
    return db


@pytest.fixture(autouse=True)
def patch_singletons(monkeypatch):
    """Patch get_predictor and get_db before importing the app."""
    predictor = _mock_predictor()
    db        = _mock_db()
    monkeypatch.setattr("api.main._predictor", predictor)
    monkeypatch.setattr("api.main._db",        db)
    # Also patch fetch so network is never hit
    monkeypatch.setattr("src.retrieval.news_fetcher.fetch_verified_news",
                        MagicMock(return_value=[]), raising=False)
    return {"predictor": predictor, "db": db}


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


# ── health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "models_ready" in data

    def test_version_present(self, client):
        r = client.get("/api/health")
        assert "version" in r.json()


# ── models ───────────────────────────────────────────────────────────────────

class TestModels:
    def test_returns_list(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert len(data["models"]) >= 1

    def test_ensemble_present(self, client):
        r = client.get("/api/models")
        keys = [m["key"] for m in r.json()["models"]]
        assert "ensemble" in keys


# ── analyze ──────────────────────────────────────────────────────────────────

class TestAnalyze:
    SAMPLE = "Scientists have discovered a new exoplanet in a nearby star system."

    def test_200_on_valid_text(self, client):
        r = client.post("/api/analyze", json={"text": self.SAMPLE})
        assert r.status_code == 200

    def test_response_has_required_keys(self, client):
        r = client.post("/api/analyze", json={"text": self.SAMPLE})
        d = r.json()
        for key in ("label", "prob_fake", "prob_real", "confidence", "model_key",
                    "threshold", "explanation", "articles", "query_id"):
            assert key in d, f"missing key: {key}"

    def test_probabilities_sum_to_one(self, client):
        r = client.post("/api/analyze", json={"text": self.SAMPLE})
        d = r.json()
        assert abs(d["prob_fake"] + d["prob_real"] - 1.0) < 0.01

    def test_explanation_is_list(self, client):
        r = client.post("/api/analyze", json={"text": self.SAMPLE})
        assert isinstance(r.json()["explanation"], list)

    def test_400_on_short_text(self, client):
        r = client.post("/api/analyze", json={"text": "hi"})
        assert r.status_code in (400, 422)

    def test_model_key_selection(self, client, patch_singletons):
        client.post("/api/analyze", json={"text": self.SAMPLE, "model_key": "lr"})
        patch_singletons["predictor"].predict.assert_called_once()
        _, kwargs = patch_singletons["predictor"].predict.call_args
        # model_key may be positional or kwarg
        called_key = kwargs.get("model_key") or patch_singletons["predictor"].predict.call_args[0][1]
        assert called_key == "lr"


# ── result ───────────────────────────────────────────────────────────────────

class TestResult:
    def test_get_existing(self, client):
        r = client.get("/api/result/42")
        assert r.status_code == 200
        d = r.json()
        assert d["query_id"] == 42
        assert "label" in d

    def test_404_on_missing(self, client, patch_singletons):
        patch_singletons["db"].get_query_by_id.return_value = None
        r = client.get("/api/result/9999")
        assert r.status_code == 404


# ── feedback ─────────────────────────────────────────────────────────────────

class TestFeedback:
    def test_yes_feedback(self, client):
        r = client.post("/api/feedback", json={"query_id": 42, "feedback": "yes"})
        assert r.status_code == 200

    def test_no_feedback(self, client):
        r = client.post("/api/feedback", json={"query_id": 42, "feedback": "no"})
        assert r.status_code == 200

    def test_invalid_feedback_rejected(self, client):
        r = client.post("/api/feedback", json={"query_id": 42, "feedback": "maybe"})
        assert r.status_code == 422


# ── history ──────────────────────────────────────────────────────────────────

class TestHistory:
    def test_returns_history_key(self, client):
        r = client.get("/api/history")
        assert r.status_code == 200
        assert "history" in r.json()

    def test_respects_limit_param(self, client, patch_singletons):
        client.get("/api/history?limit=10")
        patch_singletons["db"].get_recent_queries.assert_called_with(limit=10)
