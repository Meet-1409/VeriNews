# Project Structure

VeriNews is intentionally organized around the modules used by the current local Streamlit runtime.

```text
VeriNews/
├── app.py
├── config/
│   └── settings.py
├── data/
│   ├── raw/                 # Local source datasets; excluded from Git
│   └── processed/           # Generated dataset splits; excluded from Git
├── models/                  # Local model/vectorizer artifacts and metrics
├── src/
│   ├── preprocessing/       # Text normalization
│   ├── inference/            # Active classifier inference
│   ├── explainability/       # LIME explanations
│   ├── retrieval/            # News retrieval, scraping, and similarity ranking
│   ├── research/             # Optional live evidence brief
│   └── storage/              # SQLite persistence
├── docs/                    # Documentation and screenshot guidance
├── README.md
├── TECHNICAL_REPORT.md
└── requirements.txt
```

## Runtime Boundaries

- `app.py` is the Streamlit entry point and currently coordinates the workflow directly.
- `src/inference/` owns prediction using the active serialized model and vectorizer.
- `src/explainability/` owns LIME output for the active model.
- `src/retrieval/` owns external article retrieval and local similarity ranking.
- `src/research/` contains optional external AI enrichment.
- `src/storage/` owns local SQLite reads and writes.

The repository also contains legacy `*.cpython-313.py` recovered/decompiler artifacts in some source directories. They are documented for transparency but are not part of the maintained runtime path. They should be moved to an archive or replaced with readable source only after import dependencies are confirmed.

Large datasets, local databases, secrets, and serialized model files are excluded through `.gitignore`. A public runnable release therefore needs a documented artifact distribution process.
