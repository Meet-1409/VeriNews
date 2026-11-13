"""
VeriNews inference layer.

FakeNewsPredictor wraps the ModelManager and the LIME explainer into
a single call used by the FastAPI backend.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class FakeNewsPredictor:
    """
    High-level predictor: classification + fast coefficient explanation.

    Parameters
    ----------
    model_key   : default model to use ("ensemble", "lr", "nb", "svm")
    models_dir  : override path to the models/ directory
    model_path, vectorizer_path : legacy positional params (ignored)
    """

    def __init__(
        self,
        model_key: str = "ensemble",
        models_dir: Path | None = None,
        # Legacy params kept for backward-compat
        model_path: Path | None = None,
        vectorizer_path: Path | None = None,
    ):
        self.default_model_key = model_key
        from src.models.model_manager import ModelManager
        self._manager = ModelManager(models_dir) if models_dir else ModelManager()

    # ── public API ───────────────────────────────────────────────────────

    def predict(self, text: str, model_key: str | None = None) -> dict:
        """
        Classify text and return a structured result.

        Returns
        -------
        {
            "label":       "Fake" | "Real" | "Uncertain",
            "confidence":  float,
            "prob_fake":   float,
            "prob_real":   float,
            "model_key":   str,
            "threshold":   float,
            "explanation": list[tuple[str, float]],  # (word, weight)
        }
        """
        key    = model_key or self.default_model_key
        result = self._manager.predict(text, model_key=key)
        expl   = self._fast_explain(text, key, n=12)

        return {
            "label":       result["label"],
            "confidence":  round(max(result["prob_fake"], result["prob_real"]), 4),
            "prob_fake":   result["prob_fake"],
            "prob_real":   result["prob_real"],
            "model_key":   key,
            "threshold":   result["threshold"],
            "explanation": expl,
        }

    def explain_lime(
        self,
        text: str,
        model_key: str | None = None,
        num_features: int = 12,
    ) -> list[tuple[str, float]]:
        """
        Full LIME explanation (slower — call asynchronously in the API).

        Returns list of (word, weight) sorted by |weight| descending.
        Positive weight → pushes toward Real.
        Negative weight → pushes toward Fake.
        """
        from src.explainability.explainer import FakeNewsExplainer
        from src.preprocessing.text_cleaner import clean_text

        key    = model_key or self.default_model_key
        mgr    = self._manager
        vec    = mgr.get_tfidf()
        model  = mgr.get_model(key)

        def _predict_fn(texts):
            X = vec.transform([clean_text(t) for t in texts])
            return model.predict_proba(X)

        return FakeNewsExplainer().explain_with_fn(
            text, _predict_fn, num_features=num_features
        )

    def available_models(self) -> list[dict]:
        return self._manager.available_models()

    # ── internals ────────────────────────────────────────────────────────

    def _fast_explain(self, text: str, model_key: str, n: int = 12) -> list:
        """
        Coefficient-based word importance (instant, no LIME sampling).

        Works for LR and LinearSVC; silently returns [] for NB or ensemble
        since they have no coef_ attribute.
        """
        try:
            from src.preprocessing.text_cleaner import clean_text

            vec   = self._manager.get_tfidf()
            model = self._manager.get_model(model_key)

            # Unwrap CalibratedClassifierCV → base estimator
            cal_clfs = getattr(model, "calibrated_classifiers_", None)
            base_est = cal_clfs[0].estimator if cal_clfs else model

            if not hasattr(base_est, "coef_"):
                return []  # NB or ensemble → skip fast explain

            coef       = np.asarray(base_est.coef_).flatten()
            feat_names = vec.get_feature_names_out()

            X_row       = vec.transform([clean_text(text)])
            nonzero_idx = X_row.nonzero()[1]

            # coef_ for LinearSVC/LR: positive = more Real (class 1)
            pairs = [(feat_names[i], float(coef[i])) for i in nonzero_idx]
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return pairs[:n]

        except Exception as exc:
            log.debug("Fast explain skipped: %s", exc)
            return []
