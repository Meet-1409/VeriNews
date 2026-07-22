# VeriNews

> An academic Streamlit prototype for fake-news classification, explainable predictions, and related-news exploration.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

VeriNews is a local news-verification assistant built as a single-process Streamlit application. A user submits a headline or article excerpt and receives:

- a `Fake`, `Real`, or `Uncertain` model assessment;
- a confidence-like model score;
- a LIME word-level explanation;
- related coverage retrieved from NewsAPI or trusted publisher fallbacks;
- local SQLite history; and
- an optional evidence brief using OpenAI and Perplexity when configured.

The project is intentionally presented as a research/demo application. It is not a production fact-checking oracle, a multi-user service, or a FastAPI backend.

## Screenshots

Screenshots are reserved for the actual running application. See [docs/SCREENSHOTS.md](docs/SCREENSHOTS.md) for the capture checklist and placeholders.

<!-- Add reviewed screenshots here before publishing the repository publicly.
![Analysis workspace](docs/screenshots/analysis-workspace.png)
![Previous searches](docs/screenshots/previous-searches.png)
-->

## What Is Implemented

```text
Streamlit UI (app.py)
        |
        +--> NLTK cleaning --> TF-IDF vectorizer --> pickled classifier
        |                                      |
        |                                      +--> LIME explanation
        |
        +--> NewsAPI / publisher fallbacks --> TF-IDF similarity ranking
        |
        +--> optional OpenAI / Perplexity evidence brief
        |
        +--> SQLite query, article, and feedback history
```

The active inference path loads `models/fake_news_model.pkl` and `models/tfidf.pkl`. Other model artifacts in the repository are not selected by the current UI.

## Repository Map

```text
.
├── app.py                         # Streamlit entry point and page orchestration
├── config/settings.py             # Environment-driven configuration
├── data/                          # Local raw and processed dataset artifacts
├── models/                        # Local inference artifacts and metrics
├── src/
│   ├── preprocessing/             # Text normalization
│   ├── inference/                 # Model prediction
│   ├── explainability/            # LIME explanations
│   ├── retrieval/                 # News retrieval, scraping, and ranking
│   ├── research/                  # Optional live AI evidence brief
│   └── storage/                   # SQLite persistence
├── docs/                          # Screenshots and project documentation
├── TECHNICAL_REPORT.md            # Implementation-focused architecture report
└── requirements.txt               # Runtime dependency lower bounds
```

Some legacy `*.cpython-313.py` files remain under `src/`. They are decompiler/recovered artifacts rather than the maintained runtime path and should not be interpreted as complete implementations.

## Quick Start

### Requirements

- Python 3.11 or newer is recommended.
- The active model artifacts must be available at `models/fake_news_model.pkl` and `models/tfidf.pkl`.
- NLTK may need to download `stopwords` and `wordnet` on first startup.

### Install

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configure optional integrations

Create a local `.env` file and add only the providers you intend to use:

```dotenv
NEWS_API_KEY=
OPENAI_API_KEY=
PERPLEXITY_API_KEY=
OPENAI_MODEL=gpt-4o-mini
PERPLEXITY_MODEL=sonar
```

Submitted text may be sent to NewsAPI, OpenAI, or Perplexity when those integrations are enabled. Do not submit confidential or personally identifiable content to an unreviewed deployment.

### Run

```bash
streamlit run app.py
```

The application opens in a browser and stores local history in `verinews.db`. The database is intentionally ignored by Git.

## Model Behavior

The runtime cleans input with NLTK, transforms it using the saved TF-IDF vectorizer, and queries the saved classifier. For classifiers exposing probabilities, the current label policy is:

| Condition | Label |
| --- | --- |
| Fake probability `>= 0.90` | `Fake` |
| Otherwise, real probability `>= 0.22` | `Real` |
| Otherwise | `Uncertain` |

These are application thresholds, not calibrated truth probabilities. Results should be treated as model signals and reviewed alongside the retrieved coverage. The model is not designed to establish that a claim is true or false.

## External Services

| Service | Role | Required? |
| --- | --- | --- |
| NewsAPI | Related article retrieval | No; publisher fallbacks are used |
| OpenAI | Optional evidence-summary refinement | No |
| Perplexity | Optional research enrichment | No |
| Google Fact Check Tools | Separate retrieval utility | Not wired into the main Streamlit flow |

The application has no internal REST API, authentication, user accounts, role model, or multi-user persistence layer.

## Data and Artifacts

The repository includes dataset and model references for an academic workflow involving ISOT, LIAR, and FakeNewsNet-style sources. The active runtime artifacts are local serialized files. Because serialized model files and large datasets are excluded by `.gitignore`, a fresh clone may require a separate artifact distribution step.

Before sharing a runnable release, document:

- the exact model/vectorizer pair;
- artifact source and checksum;
- Python and dependency versions;
- dataset provenance and licenses;
- preprocessing and split configuration; and
- the evaluation run corresponding to the active model.

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the current implementation assessment and known limitations.

## Validation Status

There are currently no automated tests or CI workflows in the repository. Validation should therefore be described as manual unless tests are added. The most useful future validation targets are preprocessing behavior, threshold mapping, ranking, database persistence, and mocked external API failure paths.

## Known Limitations

- The application is single-process and synchronous.
- SQLite persistence is intended for local use, not concurrent production traffic.
- LIME explanations can be slow for long inputs.
- Retrieved publisher coverage is related evidence, not proof that the submitted claim is true.
- Static fallback links are general publisher links when live retrieval is unavailable.
- Optional AI output can be unavailable, costly, or incorrect and must be reviewed.
- Serialized model artifacts must come from a trusted source.
- No authentication, authorization, retention policy, or privacy control is implemented.

## Documentation

- [Technical report](TECHNICAL_REPORT.md): architecture, runtime flow, schema, integrations, and implementation gaps.
- [Recruiter review](RECRUITER_REVIEW.md): critical hiring-style assessment and prioritized improvements.
- [Project structure](docs/PROJECT_STRUCTURE.md): module boundaries, artifact policy, and source-tree guidance.
- [Screenshot guide](docs/SCREENSHOTS.md): placeholders and capture checklist for the GitHub page.
- [Contributing guide](CONTRIBUTING.md): documentation and local-development expectations.

## License

The application source is released under the [MIT License](LICENSE). Third-party datasets, publisher content, model artifacts, fonts, and external services may have separate terms and are not automatically covered by this license.

## Suggested GitHub Metadata

**Repository description**

> Academic Streamlit prototype for explainable fake-news classification, related-news retrieval, and optional LLM-assisted evidence summaries.

**Suggested topics**

`fake-news-detection` `machine-learning` `nlp` `explainable-ai` `lime` `tfidf` `streamlit` `scikit-learn` `news-verification` `sqlite` `python`
