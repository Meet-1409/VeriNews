"""
VeriNews similarity ranker.

Ranks news articles by TF-IDF cosine similarity to the submitted text.
Accepts either a path to a pickled vectorizer or a preloaded one to avoid
re-loading the file on every call.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)


def rank_articles_by_similarity(
    query_text: str,
    articles: list[dict],
    vectorizer_path: str | None = None,
    vectorizer=None,
    top_k: int = 5,
) -> list[dict]:
    """
    Rank articles by cosine similarity to query_text.

    Pass either `vectorizer_path` (string path to .pkl file) or a
    preloaded `vectorizer` object — the latter skips disk I/O.

    Returns the top_k articles sorted by similarity (highest first),
    with a "similarity_score" key added (raw cosine value 0–1).
    """
    if not articles or not query_text:
        logger.warning("rank_articles_by_similarity: empty input — returning as-is")
        return articles[:top_k]

    # ── load vectorizer ──────────────────────────────────────────────────
    if vectorizer is None:
        if vectorizer_path is None:
            raise ValueError("Either vectorizer_path or vectorizer must be provided")
        import pickle
        p = Path(vectorizer_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent / p
        if not p.exists():
            logger.warning("TF-IDF file not found at %s — returning unsorted", p)
            for a in articles[:top_k]:
                a.setdefault("similarity_score", 0.0)
            return articles[:top_k]
        with open(p, "rb") as f:
            vectorizer = pickle.load(f)

    # ── vectorise ────────────────────────────────────────────────────────
    cleaned_query = clean_text(query_text)

    article_texts: list[str] = []
    valid_articles: list[dict] = []
    for art in articles:
        body = art.get("description") or art.get("title") or ""
        if body.strip():
            article_texts.append(clean_text(body))
            valid_articles.append(art)

    if not article_texts:
        for a in articles[:top_k]:
            a.setdefault("similarity_score", 0.0)
        return articles[:top_k]

    try:
        query_vec   = vectorizer.transform([cleaned_query])
        article_vec = vectorizer.transform(article_texts)
        raw_sims    = cosine_similarity(query_vec, article_vec)[0]
    except Exception as exc:
        logger.warning("Similarity computation failed: %s", exc)
        for a in valid_articles[:top_k]:
            a.setdefault("similarity_score", 0.0)
        return valid_articles[:top_k]

    # ── rank & return ────────────────────────────────────────────────────
    top_indices = np.argsort(raw_sims)[::-1][:top_k]
    ranked: list[dict] = []
    for idx in top_indices:
        art = valid_articles[idx].copy()
        art["similarity_score"] = round(float(np.clip(raw_sims[idx], 0.0, 1.0)), 3)
        ranked.append(art)

    logger.info("Ranked %d articles (top score: %.3f)", len(ranked),
                ranked[0]["similarity_score"] if ranked else 0)
    return ranked
