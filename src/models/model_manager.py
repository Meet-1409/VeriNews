"""
VeriNews model registry.

Lazy-loads pickled classifiers from models/ and exposes a uniform
predict() interface keyed by short name ("lr", "nb", "svm", "ensemble").

Usage:
    from src.models.model_manager import get_manager

    mgr    = get_manager()
    result = mgr.predict("Some news headline", model_key="ensemble")
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"

# ── metadata for every model key ────────────────────────────────────────

MODEL_META: dict[str, dict] = {
    "ensemble": {
        "display_name": "Ensemble (recommended)",
        "description":  "Soft-voting average of LR, NB and SVM. Best overall accuracy.",
        "file":         "ensemble.pkl",
    },
    "lr": {
        "display_name": "Logistic Regression",
        "description":  "Fast, interpretable linear model. Great for clean, structured text.",
        "file":         "lr.pkl",
    },
    "nb": {
        "display_name": "Naive Bayes",
        "description":  "Fastest classifier. Works well on short headlines.",
        "file":         "nb.pkl",
    },
    "svm": {
        "display_name": "Linear SVM",
        "description":  "High-margin linear classifier. Excellent precision on longer articles.",
        "file":         "svm.pkl",
    },
}


class ModelManager:
    """
    Thread-safe (single-process) lazy loader for all VeriNews classifiers.

    Models are loaded on first access and cached for the process lifetime.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self._dir     = Path(models_dir) if models_dir else MODELS_DIR
        self._tfidf   = None
        self._models: dict[str, object]  = {}
        self._metrics: dict[str, dict]   = {}
        self._load_metrics()

    # ── private helpers ─────────────────────────────────────────────────

    def _load_metrics(self) -> None:
        path = self._dir / "model_metrics.json"
        if path.exists():
            with open(path) as f:
                self._metrics = json.load(f)

    def _load_tfidf(self):
        if self._tfidf is None:
            path = self._dir / "tfidf.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"TF-IDF vectorizer not found at {path}. "
                    "Run  python scripts/train_models.py  first."
                )
            with open(path, "rb") as f:
                self._tfidf = pickle.load(f)
            log.info("TF-IDF vectorizer loaded (%s features)", len(self._tfidf.vocabulary_))
        return self._tfidf

    def _load_model(self, key: str):
        if key not in self._models:
            if key not in MODEL_META:
                raise ValueError(
                    f"Unknown model key '{key}'. Choose from: {list(MODEL_META)}"
                )
            fname = MODEL_META[key]["file"]
            path  = self._dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file '{fname}' not found in {self._dir}. "
                    "Run  python scripts/train_models.py  first."
                )
            with open(path, "rb") as f:
                self._models[key] = pickle.load(f)
            log.info("Loaded model: %s", key)
        return self._models[key]

    # ── public API ───────────────────────────────────────────────────────

    def available_models(self) -> list[dict]:
        """Return model info dicts for GET /api/models."""
        result = []
        for key, meta in MODEL_META.items():
            available = (self._dir / meta["file"]).exists()
            m = self._metrics.get(key, {})
            result.append(
                {
                    "key":          key,
                    "display_name": meta["display_name"],
                    "description":  meta["description"],
                    "available":    available,
                    "metrics": {
                        "accuracy":  round(m.get("test_accuracy", 0), 3),
                        "macro_f1":  round(m.get("macro_f1",      0), 3),
                        "auc":       round(m.get("auc",           0), 3),
                        "threshold": round(m.get("threshold",     0.5), 3),
                    } if m else {},
                }
            )
        return result

    def predict(
        self,
        text: str,
        model_key: str = "ensemble",
        clean_fn=None,
    ) -> dict:
        """
        Classify a piece of text.

        Returns
        -------
        {
          "label":     "Fake" | "Real" | "Uncertain",
          "prob_fake": float,
          "prob_real": float,
          "model_key": str,
          "threshold": float,
        }
        """
        from src.preprocessing.text_cleaner import clean_text as _clean
        clean_fn = clean_fn or _clean

        vec   = self._load_tfidf()
        model = self._load_model(model_key)

        cleaned   = clean_fn(text)
        X         = vec.transform([cleaned])
        proba     = model.predict_proba(X)[0]
        prob_fake = float(proba[0])
        prob_real = float(proba[1])

        # Use validation-tuned threshold, fall back to 0.5
        threshold = float(self._metrics.get(model_key, {}).get("threshold", 0.5))

        if prob_fake >= threshold:
            label = "Fake"
        elif prob_real >= threshold:
            label = "Real"
        else:
            label = "Uncertain"

        return {
            "label":     label,
            "prob_fake": round(prob_fake, 4),
            "prob_real": round(prob_real, 4),
            "model_key": model_key,
            "threshold": round(threshold, 4),
        }

    def get_tfidf(self):
        """Expose the loaded vectorizer (for similarity ranking)."""
        return self._load_tfidf()

    def get_model(self, key: str):
        """Expose a loaded model (for LIME explanation)."""
        return self._load_model(key)


# ── module-level singleton ───────────────────────────────────────────────

_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Return the shared ModelManager, creating it on first call."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
