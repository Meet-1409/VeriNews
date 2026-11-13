"""Abstract base class for VeriNews classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseNewsModel(ABC):
    """Minimal interface every model in the registry should follow."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable key (e.g. 'lr', 'nb')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the UI."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown in the model picker."""
        ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Return (n_samples, 2) array.
        Column 0 = P(Fake), column 1 = P(Real).
        """
        ...

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
