"""
VeriNews soft-voting ensemble.

SoftEnsemble is defined at module-level so it is fully picklable
(pickle requires the class to be importable by dotted path).
"""
from __future__ import annotations
import numpy as np


class SoftEnsemble:
    """
    Soft-voting average over a list of fitted probability estimators.

    Each estimator must expose  predict_proba(X) -> ndarray(n, 2)
    where column 0 = P(Fake) and column 1 = P(Real).

    Parameters
    ----------
    estimators : list
        Pre-fitted calibrated classifiers.
    weights : list[float] | None
        Per-estimator weights (uniform if None).
    """

    def __init__(self, estimators: list, weights: list[float] | None = None):
        self.estimators = estimators
        self.weights = weights
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        probas = np.array([e.predict_proba(X) for e in self.estimators])
        if self.weights is None:
            return probas.mean(axis=0)
        w = np.array(self.weights, dtype=float)
        w = w / w.sum()
        return (probas * w[:, None, None]).sum(axis=0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
