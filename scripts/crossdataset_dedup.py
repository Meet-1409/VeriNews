#!/usr/bin/env python3
"""
Cross-dataset deduplication utility for VeriNews.

Background
----------
WELFake (72 k rows) was assembled by its authors from four existing datasets:
PolitiFact, GossipCop, BuzzFeed, and — crucially — ISOT.  About 44 k rows in
WELFake are therefore near-identical copies of ISOT rows.  Training on both
without deduplication would:
  • inflate the effective dataset size, making accuracy numbers look better
    than they really are (data leakage across the split boundary is possible);
  • bias the model towards the ISOT text style (Reuters wire copy) rather
    than learning to generalise across styles.

This script loads any combination of the five VeriNews source datasets,
reports the cross-dataset exact-duplicate picture, and writes a deduplicated
combined CSV that is suitable for training.

It is intentionally standalone: it imports nothing from the rest of the
VeriNews codebase and can be run on any machine that has the raw data files.

Usage
-----
    # dry-run: report overlap only, write nothing
    python scripts/crossdataset_dedup.py --dry-run

    # full run: load all available datasets, deduplicate, write to data/processed/
    python scripts/crossdataset_dedup.py

    # choose output location
    python scripts/crossdataset_dedup.py --out-dir /tmp/verinews_clean
"""

from __future__ import annotations
import argparse
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

ROOT = Path(__file__).resolve().parent.parent


# ── loaders (self-contained copies, no import from prepare_data) ─────────────

def _load_isot() -> pd.DataFrame:
    true_p = ROOT / "data" / "raw" / "isot" / "True.csv"
    fake_p = ROOT / "data" / "raw" / "isot" / "Fake.csv"
    if not true_p.exists() or not fake_p.exists():
        log.warning("ISOT not found at data/raw/isot/ — skipping.")
        return pd.DataFrame(columns=["text", "label", "source"])
    real_df = pd.read_csv(true_p)
    fake_df = pd.read_csv(fake_p)
    for df_ in (real_df, fake_df):
        df_["text"] = (
            df_["title"].fillna("").str.strip() + " " + df_["text"].fillna("").str.strip()
        ).str.strip()
    real_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([real_df[["text", "label"]], fake_df[["text", "label"]]], ignore_index=True)
    df["source"] = "isot"
    log.info("ISOT          : %6d rows", len(df))
    return df


def _load_welfake() -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.warning("'datasets' library not installed — skipping WELFake.")
        return pd.DataFrame(columns=["text", "label", "source"])
    try:
        ds = load_dataset("davanstrien/WELFake", trust_remote_code=True)
    except Exception as exc:
        log.warning("Could not load WELFake: %s — skipping.", exc)
        return pd.DataFrame(columns=["text", "label", "source"])
    split_key = "train" if "train" in ds else list(ds.keys())[0]
    rows = []
    for item in ds[split_key]:
        title = str(item.get("title", "")).strip()
        body  = str(item.get("text",  "")).strip()
        text  = (title + " " + body).strip()
        lbl   = int(item.get("label", -1))
        if lbl in (0, 1) and len(text) >= 50:
            rows.append({"text": text, "label": lbl, "source": "welfake"})
    df = pd.DataFrame(rows)
    log.info("WELFake       : %6d rows", len(df))
    return df


def _load_liar() -> pd.DataFrame:
    label_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("liar")
        rows = []
        for split in ds:
            for item in ds[split]:
                lbl = label_map.get(int(item.get("label", item.get("label_freeform", -1))), -1)
                if lbl == -1:
                    continue
                text = " ".join(filter(None, [
                    str(item.get("statement", "")).strip(),
                    str(item.get("subject",   "")).strip(),
                    str(item.get("context",   "")).strip(),
                ]))
                if len(text) >= 30:
                    rows.append({"text": text, "label": lbl, "source": "liar"})
        if rows:
            df = pd.DataFrame(rows)
            log.info("LIAR          : %6d rows", len(df))
            return df
    except Exception as exc:
        log.info("HuggingFace LIAR failed (%s) — skipping.", exc)
    return pd.DataFrame(columns=["text", "label", "source"])


def _load_covid() -> pd.DataFrame:
    base = ROOT / "archive-2"
    if not base.exists():
        log.warning("archive-2/ not found — skipping COVID dataset.")
        return pd.DataFrame(columns=["text", "label", "source"])
    rows = []
    for f in sorted(base.glob("*.csv")):
        name = f.stem
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
        for _, row in tmp.iterrows():
            content = str(row.get("content", "")).strip()
            title   = str(row.get("title",   "")).strip()
            ntitle  = str(row.get("newstitle", "")).strip()
            text = content if len(content) >= 50 else (title + " " + ntitle).strip()
            if len(text) >= 50:
                rows.append({"text": text, "label": lbl, "source": "covid"})
    df = pd.DataFrame(rows)
    log.info("COVID         : %6d rows", len(df))
    return df


def _load_fakenewscorpus() -> pd.DataFrame:
    path = ROOT / "fake.csv"
    if not path.exists():
        log.warning("fake.csv not found — skipping FakeNewsCorpus.")
        return pd.DataFrame(columns=["text", "label", "source"])
    try:
        raw = pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        log.warning("Could not read fake.csv: %s — skipping.", exc)
        return pd.DataFrame(columns=["text", "label", "source"])
    rows = []
    for _, row in raw.iterrows():
        title = str(row.get("title", "")).strip()
        body  = str(row.get("text",  "")).strip()
        text  = (title + " " + body).strip()
        if len(text) >= 50:
            rows.append({"text": text, "label": 0, "source": "fakenewscorpus"})
    df = pd.DataFrame(rows)
    log.info("FakeNewsCorpus: %6d rows", len(df))
    return df


# ── overlap analysis ──────────────────────────────────────────────────────────

def analyse_overlap(combined: pd.DataFrame) -> None:
    """Print a cross-dataset duplicate matrix."""
    sources = combined["source"].unique()
    log.info("")
    log.info("Cross-dataset exact-duplicate analysis")
    log.info("=" * 60)

    # Build a text→first-source map (first occurrence wins)
    text_to_source: dict[str, str] = {}
    for _, row in combined.iterrows():
        t = row["text"]
        s = row["source"]
        if t not in text_to_source:
            text_to_source[t] = s

    # Count how many rows in each source are duplicates of an EARLIER source
    dup_counts: dict[tuple[str, str], int] = {}
    seen: set[str] = set()
    for _, row in combined.iterrows():
        t = row["text"]
        s = row["source"]
        if t in seen:
            orig = text_to_source[t]
            if orig != s:
                key = (orig, s)
                dup_counts[key] = dup_counts.get(key, 0) + 1
        else:
            seen.add(t)

    if not dup_counts:
        log.info("No cross-dataset duplicates found.")
    else:
        for (src_a, src_b), count in sorted(dup_counts.items(), key=lambda x: -x[1]):
            log.info("  %-20s ← %-20s  %d duplicates", src_b, src_a, count)

    total_dups = sum(dup_counts.values())
    log.info("")
    log.info("Total cross-dataset duplicates : %d", total_dups)
    log.info("Rows after dedup               : %d", len(seen))
    log.info("")


# ── deduplication ─────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact-text duplicates, keeping the first occurrence.

    Source priority (first loaded wins):
      ISOT → LIAR → WELFake → COVID → FakeNewsCorpus

    This means WELFake rows that are copies of ISOT rows are dropped,
    not the other way around, preserving the cleaner Reuters source text.
    """
    before = len(df)
    # Enforce source priority by stable sort before drop_duplicates
    priority = {"isot": 0, "liar": 1, "welfake": 2, "covid": 3, "fakenewscorpus": 4}
    df = df.copy()
    df["_pri"] = df["source"].map(priority).fillna(99)
    df = df.sort_values("_pri").drop(columns="_pri")
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    log.info("Dedup: %d → %d  (removed %d)", before, len(df), before - len(df))
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset deduplication for VeriNews")
    p.add_argument("--dry-run",  action="store_true",
                   help="Report overlap only; do not write output files")
    p.add_argument("--out-dir",  default=str(ROOT / "data" / "processed"),
                   help="Directory to write deduplicated CSV (default: data/processed/)")
    p.add_argument("--filename", default="combined_deduped.csv",
                   help="Output filename (default: combined_deduped.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Loading datasets …")
    loaders = [_load_isot, _load_liar, _load_welfake, _load_covid, _load_fakenewscorpus]
    frames = [loader() for loader in loaders]
    frames = [f for f in frames if len(f) > 0]

    if not frames:
        log.error("No datasets loaded. Check that raw data files are in place.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Total before dedup : %d rows", len(combined))

    # Length filter (matches prepare_data.py)
    combined = combined[(combined["text"].str.len() >= 50) & (combined["text"].str.len() <= 20_000)]

    analyse_overlap(combined)

    if args.dry_run:
        log.info("--dry-run: no files written.")
        return

    deduped = deduplicate(combined)

    dist = deduped["label"].value_counts().to_dict()
    log.info("Final corpus: %d rows — Fake=%d  Real=%d",
             len(deduped), dist.get(0, 0), dist.get(1, 0))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename
    deduped.to_csv(out_path, index=False)
    log.info("Written to %s", out_path)

    # Summary per source after dedup
    log.info("")
    log.info("Rows per source after dedup:")
    for src, cnt in deduped["source"].value_counts().items():
        log.info("  %-20s : %d", src, cnt)


if __name__ == "__main__":
    main()
