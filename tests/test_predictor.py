"""
Unit tests for FakeNewsPredictor and ModelManager.

Run with:
    pytest tests/test_predictor.py -v

These tests use mocked pickled models so no training run is required.
"""
from __future__ import annotations

import os
import sys
import types
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── helpers ──────────────────────────────────────────────────────────────────

class _FakeTfidf:
    """Picklable TF-IDF stub — returns a fixed sparse-like array."""
    vocabulary_ = {f"word{i}": i for i in range(10)}

    def transform(self, texts):
        import scipy.sparse as sp
        n = len(texts) if hasattr(texts, "__len__") else 1
        return sp.csr_matrix(np.zeros((n, 10)))


class _FakeClassifier:
    """Picklable classifier stub with fixed predict_proba."""
    def __init__(self, prob_fake: float = 0.3):
        self._prob_fake = prob_fake
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else 1
        return np.array([[self._prob_fake, 1.0 - self._prob_fake]] * n)


def _make_fake_tfidf():
    return _FakeTfidf()


def _make_fake_classifier(prob_fake: float = 0.3):
    return _FakeClassifier(prob_fake)


def _write_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Create a temp model directory with fake .pkl files and metrics JSON."""
    tfidf = _make_fake_tfidf()
    ensemble = _make_fake_classifier(0.2)
    lr      = _make_fake_classifier(0.25)
    nb      = _make_fake_classifier(0.35)
    svm     = _make_fake_classifier(0.30)

    _write_pickle(tfidf,    tmp_path / "tfidf.pkl")
    _write_pickle(ensemble, tmp_path / "ensemble.pkl")
    _write_pickle(lr,       tmp_path / "lr.pkl")
    _write_pickle(nb,       tmp_path / "nb.pkl")
    _write_pickle(svm,      tmp_path / "svm.pkl")

    import json
    metrics = {
        "ensemble": {"threshold": 0.5, "accuracy": 0.95, "f1_macro": 0.94},
        "lr":       {"threshold": 0.5, "accuracy": 0.93, "f1_macro": 0.92},
        "nb":       {"threshold": 0.5, "accuracy": 0.90, "f1_macro": 0.89},
        "svm":      {"threshold": 0.5, "accuracy": 0.92, "f1_macro": 0.91},
    }
    (tmp_path / "model_metrics.json").write_text(json.dumps(metrics))

    return tmp_path


# ── ModelManager tests ────────────────────────────────────────────────────────

class TestModelManager:
    def test_available_models_lists_four(self, model_dir):
        from src.models.model_manager import ModelManager
        mgr = ModelManager(models_dir=str(model_dir))
        models = mgr.available_models()
        assert len(models) == 4
        keys = {m["key"] for m in models}
        assert keys == {"ensemble", "lr", "nb", "svm"}

    def test_all_marked_available(self, model_dir):
        from src.models.model_manager import ModelManager
        mgr = ModelManager(models_dir=str(model_dir))
        for m in mgr.available_models():
            assert m["available"] is True

    def test_predict_returns_expected_keys(self, model_dir):
        from src.models.model_manager import ModelManager
        mgr = ModelManager(models_dir=str(model_dir))
        result = mgr.predict("Some news text to classify", "ensemble", clean_fn=str)
        assert "label"      in result
        assert "prob_fake"  in result
        assert "prob_real"  in result
        assert "model_key"  in result
        assert "threshold"  in result

    def test_label_real_when_prob_fake_low(self, model_dir):
        from src.models.model_manager import ModelManager
        mgr = ModelManager(models_dir=str(model_dir))
        # ensemble is set to prob_fake=0.2, threshold=0.5 → label=Real
        result = mgr.predict("Legitimate news article about policy", "ensemble", clean_fn=str)
        assert result["label"] == "Real"

    def test_label_fake_when_prob_fake_high(self, tmp_path):
        """When prob_fake > threshold the label should be Fake."""
        import json, pickle
        tfidf = _make_fake_tfidf()
        fake_clf = _make_fake_classifier(prob_fake=0.8)
        _write_pickle(tfidf,    tmp_path / "tfidf.pkl")
        _write_pickle(fake_clf, tmp_path / "ensemble.pkl")
        metrics = {"ensemble": {"threshold": 0.5}}
        (tmp_path / "model_metrics.json").write_text(json.dumps(metrics))

        from src.models.model_manager import ModelManager
        mgr = ModelManager(models_dir=str(tmp_path))
        result = mgr.predict("Fake news headline with wild claims", "ensemble", clean_fn=str)
        assert result["label"] == "Fake"


# ── FakeNewsPredictor tests ────────────────────────────────────────────────────

class TestFakeNewsPredictor:
    def test_predict_shape(self, model_dir):
        from src.inference.predictor import FakeNewsPredictor
        p = FakeNewsPredictor(models_dir=str(model_dir))
        r = p.predict("Breaking: Scientists discover new planet in solar system")
        assert r["label"]      in {"Fake", "Real", "Uncertain"}
        assert 0.0 <= r["prob_fake"] <= 1.0
        assert 0.0 <= r["prob_real"] <= 1.0
        assert abs(r["prob_fake"] + r["prob_real"] - 1.0) < 0.01

    def test_model_key_echoed(self, model_dir):
        from src.inference.predictor import FakeNewsPredictor
        p = FakeNewsPredictor(models_dir=str(model_dir))
        r = p.predict("Test claim", model_key="lr")
        assert r["model_key"] == "lr"

    def test_confidence_in_range(self, model_dir):
        from src.inference.predictor import FakeNewsPredictor
        p = FakeNewsPredictor(models_dir=str(model_dir))
        r = p.predict("Government issues health advisory", model_key="nb")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_explanation_is_list(self, model_dir):
        from src.inference.predictor import FakeNewsPredictor
        p = FakeNewsPredictor(models_dir=str(model_dir))
        r = p.predict("Health experts warn about flu season")
        assert isinstance(r.get("explanation", []), list)

    def test_available_models_passthrough(self, model_dir):
        from src.inference.predictor import FakeNewsPredictor
        p = FakeNewsPredictor(models_dir=str(model_dir))
        models = p.available_models()
        assert len(models) >= 1
