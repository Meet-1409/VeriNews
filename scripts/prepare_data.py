#!/usr/bin/env python3
"""
VeriNews dataset preparation.

Downloads and merges (combined ≈130–160 k samples):
  ISOT     – 44,898 rows — requires MANUAL Kaggle download (see below)
  LIAR     – 12,836 rows — auto-downloaded via HuggingFace datasets
  WELFake  – 72,134 rows — auto-downloaded via HuggingFace datasets

Label encoding:  0 = Fake   1 = Real

ISOT manual step (do this first!):
    1. Visit https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    2. Download and unzip
    3. Place  True.csv  and  Fake.csv  in  data/raw/isot/

Then run:
    python scripts/prepare_data.py

Outputs in data/processed/:
    combined_dataset.csv   train.csv   validation.csv   test.csv
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

for d in [RAW / "isot", RAW / "liar", RAW / "welfake", PROC]:
    d.mkdir(parents=True, exist_ok=True)


# ── individual loaders ───────────────────────────────────────────────────

def load_isot() -> pd.DataFrame:
    true_p = RAW / "isot" / "True.csv"
    fake_p = RAW / "isot" / "Fake.csv"

    if not true_p.exists() or not fake_p.exists():
        log.warning(
            "ISOT not found in data/raw/isot/ — skipping.\n"
            "  → Download from https://www.kaggle.com/datasets/"
            "clmentbisaillon/fake-and-real-news-dataset\n"
            "  → Place True.csv + Fake.csv in data/raw/isot/"
        )
        return pd.DataFrame(columns=["text", "label", "source"])

    real_df = pd.read_csv(true_p)
    fake_df = pd.read_csv(fake_p)

    # ISOT columns: title, text, subject, date — concatenate title + body
    for df_ in (real_df, fake_df):
        df_["text"] = (
            df_["title"].fillna("").str.strip()
            + " "
            + df_["text"].fillna("").str.strip()
        ).str.strip()

    real_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat(
        [real_df[["text", "label"]], fake_df[["text", "label"]]],
        ignore_index=True,
    )
    df["source"] = "isot"
    log.info("ISOT loaded: %d rows", len(df))
    return df


def load_liar() -> pd.DataFrame:
    """
    Load LIAR dataset. Tries three strategies in order:
      1. HuggingFace datasets (no trust_remote_code — works if Parquet version exists)
      2. Direct TSV download from GitHub (always works, no HF dependency)
      3. Skip gracefully
    """
    # LIAR fine-grained labels:
    #   0=pants-fire  1=false  2=barely-true → Fake (0)
    #   3=half-true   4=mostly-true  5=true  → Real (1)
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    # ── Strategy 1: HuggingFace datasets (Parquet-only, no script) ──────────
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("liar")          # official HF dataset
        rows: list[dict] = []
        for split in ds:
            for item in ds[split]:
                raw_lbl = item.get("label", item.get("label_freeform", -1))
                lbl = label_map.get(int(raw_lbl), -1)
                if lbl == -1:
                    continue
                stmt    = str(item.get("statement", "")).strip()
                context = str(item.get("context",   "")).strip()
                subject = str(item.get("subject",   "")).strip()
                text    = " ".join(filter(None, [stmt, subject, context]))
                if len(text) >= 30:
                    rows.append({"text": text, "label": lbl, "source": "liar"})
        if rows:
            df = pd.DataFrame(rows)
            log.info("LIAR loaded via HuggingFace: %d rows", len(df))
            return df
    except Exception as exc:
        log.info("HuggingFace LIAR failed (%s) — trying direct TSV download.", exc)

    # ── Strategy 2: Direct GitHub TSV download ───────────────────────────────
    import io, zipfile
    import requests as req_lib

    LIAR_ZIP = "https://raw.githubusercontent.com/thiagohersan/liar-dataset/master/dataset/liar_dataset.zip"
    # Columns: id, label, statement, subject, speaker, job, state, party,
    #          barely_true_count, false_count, half_true_count, mostly_true_count,
    #          pants_on_fire_count, context
    COL_NAMES = [
        "id", "label_str", "statement", "subject", "speaker", "job_title",
        "state", "party", "barely_true", "false_ct", "half_true",
        "mostly_true", "pants_fire", "context",
    ]
    LABEL_STR_MAP = {
        "pants-fire": 0, "false": 0, "barely-true": 0,
        "half-true": 1, "mostly-true": 1, "true": 1,
    }

    try:
        log.info("Downloading LIAR TSV from GitHub …")
        resp = req_lib.get(LIAR_ZIP, timeout=60)
        resp.raise_for_status()
        rows = []
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for fname in zf.namelist():
                if fname.endswith(".tsv"):
                    with zf.open(fname) as f:
                        content = f.read().decode("utf-8", errors="replace")
                    tmp = pd.read_csv(
                        io.StringIO(content), sep="\t",
                        header=None, names=COL_NAMES[:14],
                        on_bad_lines="skip",
                    )
                    for _, row in tmp.iterrows():
                        lbl = LABEL_STR_MAP.get(str(row.get("label_str", "")).strip(), -1)
                        if lbl == -1:
                            continue
                        stmt    = str(row.get("statement", "")).strip()
                        context = str(row.get("context",   "")).strip()
                        subject = str(row.get("subject",   "")).strip()
                        text    = " ".join(filter(None, [stmt, subject, context]))
                        if len(text) >= 30:
                            rows.append({"text": text, "label": lbl, "source": "liar"})
        if rows:
            df = pd.DataFrame(rows)
            log.info("LIAR loaded via GitHub TSV: %d rows", len(df))
            return df
    except Exception as exc:
        log.warning("GitHub LIAR download failed: %s — skipping.", exc)

    return pd.DataFrame(columns=["text", "label", "source"])


def load_welfake() -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.warning("'datasets' library not installed — skipping WELFake.  pip install datasets")
        return pd.DataFrame(columns=["text", "label", "source"])

    try:
        ds = load_dataset("davanstrien/WELFake", trust_remote_code=True)
    except Exception as exc:
        log.warning("Could not download WELFake: %s — skipping.", exc)
        return pd.DataFrame(columns=["text", "label", "source"])

    # WELFake: title, text, label (0=Fake, 1=Real)
    rows: list[dict] = []
    split_key = "train" if "train" in ds else list(ds.keys())[0]
    for item in ds[split_key]:
        title = str(item.get("title", "")).strip()
        body  = str(item.get("text",  "")).strip()
        text  = (title + " " + body).strip()
        lbl   = int(item.get("label", -1))
        if lbl in (0, 1) and len(text) >= 50:
            rows.append({"text": text, "label": lbl, "source": "welfake"})

    df = pd.DataFrame(rows)
    log.info("WELFake loaded: %d rows", len(df))
    return df


def load_covid_fakenews() -> pd.DataFrame:
    """
    archive-2/ — COVID-19 fake/real news.
    Label is encoded in the filename: *Fake* → 0, *Real* → 1.
    Uses the base News files (not _tweets or _5/_7 variants) for article-length text.
    """
    base = ROOT / "archive-2"
    if not base.exists():
        log.warning("archive-2/ not found — skipping COVID dataset.")
        return pd.DataFrame(columns=["text", "label", "source"])

    rows: list[dict] = []
    for f in sorted(base.glob("*.csv")):
        name = f.stem  # e.g. NewsFakeCOVID-19
        if "Fake" in name:
            lbl = 0
        elif "Real" in name:
            lbl = 1
        else:
            continue
        try:
            tmp = pd.read_csv(f, on_bad_lines="skip")
        except Exception:
            continue
        # Prefer 'content' column; fall back to title + newstitle
        for _, row in tmp.iterrows():
            content = str(row.get("content", "")).strip()
            title   = str(row.get("title",   "")).strip()
            ntitle  = str(row.get("newstitle", "")).strip()
            text = content if len(content) >= 50 else (title + " " + ntitle).strip()
            if len(text) >= 50:
                rows.append({"text": text, "label": lbl, "source": "covid"})

    df = pd.DataFrame(rows)
    log.info("COVID fake news loaded: %d rows", len(df))
    return df


def load_fakenews_corpus() -> pd.DataFrame:
    """
    fake.csv in project root — FakeNewsCorpus subset (bs/conspiracy/bias/etc.).
    All rows are misinformation; label = 0 (Fake).
    """
    path = ROOT / "fake.csv"
    if not path.exists():
        log.warning("fake.csv not found in project root — skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])

    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        log.warning("Could not read fake.csv: %s — skipping.", exc)
        return pd.DataFrame(columns=["text", "label", "source"])

    rows: list[dict] = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")).strip()
        body  = str(row.get("text",  "")).strip()
        text  = (title + " " + body).strip()
        if len(text) >= 50:
            rows.append({"text": text, "label": 0, "source": "fakenewscorpus"})

    result = pd.DataFrame(rows)
    log.info("FakeNewsCorpus loaded: %d rows", len(result))
    return result


# ── cleaning ─────────────────────────────────────────────────────────────

def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()

    # Drop very short / very long texts
    df = df[(df["text"].str.len() >= 50) & (df["text"].str.len() <= 20_000)]

    # Remove exact duplicates (keep first)
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    log.info("Dedup: %d → %d rows  (removed %d)", before, len(df), before - len(df))

    return df.reset_index(drop=True)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading datasets …")
    frames: list[pd.DataFrame] = []

    for loader in (load_isot, load_liar, load_welfake, load_covid_fakenews, load_fakenews_corpus):
        df = loader()
        if len(df):
            frames.append(df)

    if not frames:
        log.error(
            "No datasets loaded!\n"
            "  • Install the datasets library:  pip install datasets\n"
            "  • Download ISOT from Kaggle and place in data/raw/isot/"
        )
        sys.exit(1)

    combined = clean_and_deduplicate(pd.concat(frames, ignore_index=True))

    dist = combined["label"].value_counts().to_dict()
    log.info("Combined: %d rows — Fake=%d  Real=%d", len(combined), dist.get(0, 0), dist.get(1, 0))

    # Stratified split: 70 / 15 / 15
    train, temp = train_test_split(
        combined, test_size=0.30, stratify=combined["label"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=42
    )

    combined.to_csv(PROC / "combined_dataset.csv", index=False)
    train.to_csv(   PROC / "train.csv",             index=False)
    val.to_csv(     PROC / "validation.csv",         index=False)
    test.to_csv(    PROC / "test.csv",               index=False)

    log.info("Saved — train=%d  val=%d  test=%d", len(train), len(val), len(test))
    log.info("Done. Files in data/processed/")


if __name__ == "__main__":
    main()
