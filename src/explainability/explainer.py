"""
VeriNews LIME-based word-level explanation.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)


class FakeNewsExplainer:
    """LIME text explainer with a lazy-initialised explainer instance."""

    def __init__(self):
        self._lime_explainer = None

    def _get_lime(self):
        if self._lime_explainer is None:
            from lime.lime_text import LimeTextExplainer  # type: ignore
            self._lime_explainer = LimeTextExplainer(
                class_names=["Fake", "Real"],
                random_state=42,
            )
        return self._lime_explainer

    def explain(self, text: str, num_features: int = 12) -> list:
        """Legacy stub — requires predict_fn set externally. Use explain_with_fn."""
        return []

    def explain_with_fn(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 12,
    ) -> list[tuple[str, float]]:
        """
        Run LIME with a provided predict_fn(texts) → proba array (n, 2).

        Returns
        -------
        list of (word, weight) sorted by |weight| descending.
        Positive weight → word pushes toward Real.
        Negative weight → word pushes toward Fake.
        """
        try:
            exp = self._get_lime().explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=400,   # keep fast; increase for production
                labels=[0],        # explain Fake class
            )
            # exp.as_list(label=0): (word, weight_for_fake)
            # negate so positive = Real-pushing, negative = Fake-pushing
            pairs = [(w, -wt) for w, wt in exp.as_list(label=0)]
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            return pairs
        except Exception as exc:
            log.warning("LIME explanation failed: %s", exc)
            return []
