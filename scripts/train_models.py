#!/usr/bin/env python3
"""
VeriNews multi-model training script.

Trains the following classifiers on the prepared TF-IDF features:

  lr        — Logistic Regression (C=1.0, balanced, lbfgs)
  nb        — Multinomial Naive Bayes (alpha=0.1)
  svm       — Linear SVM (CalibratedClassifierCV wrapper)
  ensemble  — Soft-voting average of the three above (default model)

Each classifier is wrapped in CalibratedClassifierCV so predict_proba()
returns well-calibrated probabilities.

A validation-set threshold search is run per model to find the
operating point with the best macro-F1 (balances fake-recall and
real-precision).

Outputs in models/:
    tfidf.pkl            — fitted TF-IDF vectorizer
    lr.pkl / nb.pkl / svm.pkl / ensemble.pkl
    fake_news_model.pkl  — alias for ensemble (backward compat)
    model_metrics.json   — per-model accuracy, F1, AUC, threshold

Run:
    python scripts/train_models.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.voting_ensemble import SoftEnsemble        # picklable
from src.preprocessing.text_cleaner import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROC   = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────

def load_split(name: str) -> tuple[list[str], np.ndarray]:
    path = PROC / f"{name}.csv"
    if not path.exists():
        log.error("Missing %s — run  python scripts/prepare_data.py  first.", path)
        sys.exit(1)
    df = pd.read_csv(path)
    texts  = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).values
    log.info("Loaded %s: %d rows", name, len(texts))
    return texts, labels


def clean_texts(texts: list[str]) -> list[str]:
    log.info("Cleaning %d texts …", len(texts))
    return [clean_text(t) for t in texts]


def find_threshold(proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    Grid-search the fake-class probability threshold that maximises macro-F1.
    proba: predict_proba output, shape (n, 2); column 0 = P(Fake).
    """
    prob_fake = proba[:, 0]
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.30, 0.80, 0.02):
        preds = np.where(prob_fake >= t, 0, 1)   # 0=Fake, 1=Real
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def evaluate(model, X_test, y_test: np.ndarray, threshold: float) -> dict:
    proba     = model.predict_proba(X_test)
    prob_fake = proba[:, 0]
    preds     = np.where(prob_fake >= threshold, 0, 1)

    auc    = float(roc_auc_score(y_test, proba[:, 1]))
    report = classification_report(
        y_test, preds,
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "test_accuracy":  round(float(accuracy_score(y_test, preds)), 4),
        "precision":      round(float(report["Real"]["precision"]),    4),
        "recall":         round(float(report["Real"]["recall"]),       4),
        "f1":             round(float(report["Real"]["f1-score"]),     4),
        "fake_precision": round(float(report["Fake"]["precision"]),    4),
        "fake_recall":    round(float(report["Fake"]["recall"]),       4),
        "fake_f1":        round(float(report["Fake"]["f1-score"]),     4),
        "macro_f1":       round(float(report["macro avg"]["f1-score"]),4),
        "auc":            round(auc,                                   4),
        "threshold":      round(threshold,                             4),
        "confusion": {
            "TP": int(((preds == 1) & (y_test == 1)).sum()),
            "FP": int(((preds == 1) & (y_test == 0)).sum()),
            "TN": int(((preds == 0) & (y_test == 0)).sum()),
            "FN": int(((preds == 0) & (y_test == 1)).sum()),
        },
    }


def save_pkl(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Saved  %s", path.name)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Data ─────────────────────────────────────────────────────────
    train_raw, y_train = load_split("train")
    val_raw,   y_val   = load_split("validation")
    test_raw,  y_test  = load_split("test")

    train_clean = clean_texts(train_raw)
    val_clean   = clean_texts(val_raw)
    test_clean  = clean_texts(test_raw)

    # ── 2. TF-IDF ───────────────────────────────────────────────────────
    log.info("Fitting TF-IDF vectorizer …")
    vec = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    X_train = vec.fit_transform(train_clean)
    X_val   = vec.transform(val_clean)
    X_test  = vec.transform(test_clean)
    save_pkl(vec, MODELS / "tfidf.pkl")

    # ── 3. Base estimators ──────────────────────────────────────────────
    log.info("Training classifiers …")

    lr_base  = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=42,
    )
    nb_base  = MultinomialNB(alpha=0.1)
    svm_base = LinearSVC(
        C=0.5, max_iter=2000,
        class_weight="balanced", random_state=42,
    )

    # Calibrate all with isotonic regression (3-fold CV on training data)
    lr  = CalibratedClassifierCV(lr_base,  cv=3, method="isotonic")
    nb  = CalibratedClassifierCV(nb_base,  cv=3, method="isotonic")
    svm = CalibratedClassifierCV(svm_base, cv=3, method="isotonic")

    lr.fit(X_train,  y_train)
    nb.fit(X_train,  y_train)
    svm.fit(X_train, y_train)

    # ── 4. Ensemble ─────────────────────────────────────────────────────
    log.info("Building soft-voting ensemble …")
    ensemble = SoftEnsemble([lr, nb, svm])

    # ── 5. Save models ──────────────────────────────────────────────────
    save_pkl(lr,       MODELS / "lr.pkl")
    save_pkl(nb,       MODELS / "nb.pkl")
    save_pkl(svm,      MODELS / "svm.pkl")
    save_pkl(ensemble, MODELS / "ensemble.pkl")
    save_pkl(ensemble, MODELS / "fake_news_model.pkl")  # backward-compat alias

    # ── 6. Threshold search on validation set ───────────────────────────
    log.info("Finding optimal thresholds on validation set …")
    named = {"lr": lr, "nb": nb, "svm": svm, "ensemble": ensemble}
    thresholds: dict[str, float] = {}
    for name, model in named.items():
        proba_val      = model.predict_proba(X_val)
        thresholds[name] = find_threshold(proba_val, y_val)
        log.info("  %s  →  threshold = %.2f", name, thresholds[name])

    # ── 7. Evaluate on test set ─────────────────────────────────────────
    log.info("Evaluating on test set …")
    all_metrics: dict[str, dict] = {}
    for name, model in named.items():
        m = evaluate(model, X_test, y_test, thresholds[name])
        all_metrics[name] = m
        log.info(
            "  %-10s  acc=%.3f  macro-F1=%.3f  AUC=%.3f  thr=%.2f",
            name, m["test_accuracy"], m["macro_f1"], m["auc"], m["threshold"],
        )

    with open(MODELS / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved model_metrics.json")

    # ── 8. Summary ──────────────────────────────────────────────────────
    best = max(all_metrics, key=lambda k: all_metrics[k]["macro_f1"])
    log.info("\n=== Training complete ===")
    log.info("Best model by macro-F1: %s (%.3f)", best, all_metrics[best]["macro_f1"])
    log.info("Models saved to models/: tfidf.pkl, lr.pkl, nb.pkl, svm.pkl, ensemble.pkl")


if __name__ == "__main__":
    main()
