#!/usr/bin/env python3
"""
DistilBERT baseline for VeriNews.

Fine-tunes distilbert-base-uncased on the SAME train/val/test split used for
the TF-IDF ensemble, so the macro-F1 numbers are directly comparable.

Requirements (install separately — not in main requirements.txt):
    pip install torch transformers accelerate

Runtime:
    CPU  : ~40–60 min
    GPU  : ~5–8 min
    Colab: recommended (free T4 is enough)

Outputs:
    models/bert_baseline/          — fine-tuned model + tokenizer
    models/bert_metrics.json       — accuracy, macro-F1, AUC on test set

Usage:
    python scripts/train_bert_baseline.py
    python scripts/train_bert_baseline.py --epochs 2 --batch-size 16
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
PROC       = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
BERT_DIR   = MODELS_DIR / "bert_baseline"


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on VeriNews data")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--max-len",    type=int,   default=256,
                   help="Max token length (256 covers ~95%% of articles)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ── dataset wrapper ───────────────────────────────────────────────────────────

def make_dataset(texts: list[str], labels: list[int], tokenizer, max_len: int):
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        raise SystemExit("PyTorch not found. Run: pip install torch transformers accelerate")

    class _DS(Dataset):
        def __init__(self):
            self.enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids":      self.enc["input_ids"][idx],
                "attention_mask": self.enc["attention_mask"][idx],
                "labels":         self.labels[idx],
            }

    return _DS()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    except ImportError as e:
        log.error("Missing dependency: %s\nRun: pip install torch transformers accelerate", e)
        sys.exit(1)

    # ── reproducibility ──────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── load splits ──────────────────────────────────────────────────────────
    for split in ("train", "validation", "test"):
        path = PROC / f"{split}.csv"
        if not path.exists():
            log.error("Missing %s — run scripts/prepare_data.py first.", path)
            sys.exit(1)

    log.info("Loading splits …")
    train_df = pd.read_csv(PROC / "train.csv")
    val_df   = pd.read_csv(PROC / "validation.csv")
    test_df  = pd.read_csv(PROC / "test.csv")

    log.info("train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    # ── tokeniser + datasets ─────────────────────────────────────────────────
    log.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log.info("Tokenising (max_len=%d) …", args.max_len)
    train_ds = make_dataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, args.max_len)
    val_ds   = make_dataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer, args.max_len)
    test_ds  = make_dataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer, args.max_len)

    # ── model ────────────────────────────────────────────────────────────────
    log.info("Loading model: %s", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # ── metrics fn ───────────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        proba = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "auc":      roc_auc_score(labels, proba),
        }

    # ── training args ────────────────────────────────────────────────────────
    BERT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BERT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── train ────────────────────────────────────────────────────────────────
    log.info("Starting fine-tuning (%d epochs, batch=%d, lr=%g) …",
             args.epochs, args.batch_size, args.lr)
    trainer.train()

    # ── evaluate on test set ─────────────────────────────────────────────────
    log.info("Evaluating on test set …")
    results = trainer.evaluate(test_ds)

    metrics = {
        "model":    args.model_name,
        "epochs":   args.epochs,
        "accuracy": round(results.get("eval_accuracy", 0), 4),
        "macro_f1": round(results.get("eval_macro_f1", 0), 4),
        "auc":      round(results.get("eval_auc", 0), 4),
        "device":   device,
    }

    out_path = MODELS_DIR / "bert_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("bert_baseline results:")
    log.info("  accuracy = %.4f", metrics["accuracy"])
    log.info("  macro-F1 = %.4f", metrics["macro_f1"])
    log.info("  AUC      = %.4f", metrics["auc"])
    log.info("Saved %s", out_path)

    # ── save model ───────────────────────────────────────────────────────────
    trainer.save_model(str(BERT_DIR))
    tokenizer.save_pretrained(str(BERT_DIR))
    log.info("Model saved to %s", BERT_DIR)

    # ── comparison against ensemble ──────────────────────────────────────────
    ensemble_metrics_path = MODELS_DIR / "model_metrics.json"
    if ensemble_metrics_path.exists():
        with open(ensemble_metrics_path) as f:
            em = json.load(f)
        ensemble_f1 = em.get("ensemble", {}).get("macro_f1", None)
        if ensemble_f1:
            diff = metrics["macro_f1"] - ensemble_f1
            log.info("")
            log.info("=== Comparison ===")
            log.info("  Ensemble macro-F1 : %.4f", ensemble_f1)
            log.info("  BERT     macro-F1 : %.4f", metrics["macro_f1"])
            log.info("  Delta            : %+.4f", diff)
            if abs(diff) <= 0.03:
                log.info("  → Within 3 pp: ensemble choice justified by interpretability.")
            elif diff > 0.03:
                log.info("  → BERT significantly better: consider updating SOP to use BERT.")
            else:
                log.info("  → Ensemble outperforms BERT: strong justification for TF-IDF approach.")


if __name__ == "__main__":
    main()
