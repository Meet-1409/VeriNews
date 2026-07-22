# VeriNews Technical Report

**Assessment date:** July 22, 2026
**Repository:** VeriNews

## Executive Summary

VeriNews is a single-process Streamlit application for fake-news checking. Its implemented runtime combines:

- a local TF-IDF vectorizer and pickled classifier for the primary prediction;
- LIME for word-level explanations;
- NewsAPI retrieval of related articles from trusted sources;
- optional OpenAI and Perplexity calls for a live evidence brief; and
- local SQLite persistence for query history, retrieved articles, and feedback.

The README describes a broader platform than the runnable code currently implements. The shipped application is not a FastAPI service, does not expose REST endpoints, does not implement authentication, and does not offer model selection in the UI. The actual runtime path is centered on [`app.py`](app.py), [`predictor.py`](src/inference/predictor.py), [`explainer.py`](src/explainability/explainer.py), [`news_fetcher.py`](src/retrieval/news_fetcher.py), [`similarity_ranker.py`](src/retrieval/similarity_ranker.py), and [`database.py`](src/storage/database.py).

## Project Purpose

The practical purpose of VeriNews is to let a user paste a headline or article excerpt and receive:

- a `Fake`, `Real`, or `Uncertain` classification;
- a confidence score;
- a LIME explanation showing influential words;
- a ranked list of related articles from trusted sources;
- saved history of prior searches; and
- optional AI-generated fact-check narration based on live evidence.

In short, VeriNews is a local news-verification assistant rather than a full multi-service production platform.

## Repository Structure

```text
VeriNews/
├─ app.py
├─ README.md
├─ requirements.txt
├─ config/
│  └─ settings.py
├─ data/
│  ├─ raw/
│  │  ├─ credibility/
│  │  ├─ fakenewsnet/
│  │  ├─ isot/
│  │  ├─ liar/
│  │  ├─ fake_news.csv
│  │  └─ real_news.csv
│  └─ processed/
│     ├─ combined_dataset.csv
│     ├─ train.csv
│     ├─ validation.csv
│     └─ test.csv
├─ models/
│  ├─ fake_news_model.pkl
│  ├─ tfidf.pkl
│  ├─ logistic_regression.pkl
│  ├─ svm.pkl
│  ├─ naive_bayes.pkl
│  ├─ bert_cnn.pkl
│  ├─ cnn_lstm.pkl / cnn_lstm.h5
│  ├─ ensemble.pkl
│  ├─ model_metrics.json
│  └─ training_monitoring.json
└─ src/
   ├─ preprocessing/text_cleaner.py
   ├─ inference/predictor.py
   ├─ explainability/explainer.py
   ├─ retrieval/news_fetcher.py
   ├─ retrieval/similarity_ranker.py
   ├─ retrieval/factcheck_api.py
   ├─ retrieval/news_scraper.py
   ├─ research/live_factcheck.py
   ├─ storage/database.py
   ├─ analysis/
   ├─ models/
   ├─ training/
   └─ utils/
```

Some directories also contain small `.cpython-313.py` decompiler-output artifacts or placeholders rather than complete, directly maintained source implementations.

## Implemented Architecture

The actual architecture is monolithic and local, with no HTTP boundary between the presentation and application logic.

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| Presentation | [`app.py`](app.py) | Streamlit interface and workflow orchestration |
| Inference | [`predictor.py`](src/inference/predictor.py) | Load the TF-IDF vectorizer and pickled classifier; produce labels and confidence |
| Explainability | [`explainer.py`](src/explainability/explainer.py) | Generate LIME feature explanations |
| Retrieval | [`news_fetcher.py`](src/retrieval/news_fetcher.py) | Retrieve related articles from NewsAPI or fallback sources |
| Ranking | [`similarity_ranker.py`](src/retrieval/similarity_ranker.py) | Rank articles with cosine similarity in TF-IDF space |
| Research augmentation | [`live_factcheck.py`](src/research/live_factcheck.py) | Build an optional AI evidence brief |
| Persistence | [`database.py`](src/storage/database.py) | Store queries, articles, and feedback in SQLite |
| Configuration | [`settings.py`](config/settings.py) | Load environment-driven keys and source configuration |

## Backend Flow

1. Streamlit starts from [`app.py`](app.py).
2. Cached resources load: `FakeNewsPredictor`, `FakeNewsExplainer`, `VeriNewsDB`, and `LiveFactChecker`.
3. The user submits a headline or article excerpt.
4. [`text_cleaner.py`](src/preprocessing/text_cleaner.py) lowercases the text, strips non-letters, removes stopwords, and lemmatizes it with NLTK.
5. [`predictor.py`](src/inference/predictor.py) vectorizes the cleaned text with `tfidf.pkl`, scores it with `fake_news_model.pkl`, and maps probabilities to labels:
   - `Fake` when fake probability is at least `0.90`;
   - `Real` when real probability is at least `0.22`;
   - `Uncertain` otherwise.
6. The query result is stored in SQLite.
7. [`explainer.py`](src/explainability/explainer.py) runs LIME against the same model and vectorizer.
8. [`news_fetcher.py`](src/retrieval/news_fetcher.py) retrieves trusted articles from NewsAPI or falls back to static Reuters, BBC, and AP links.
9. [`similarity_ranker.py`](src/retrieval/similarity_ranker.py) ranks retrieved articles using cosine similarity over the same TF-IDF space.
10. Ranked articles are saved in SQLite.
11. When OpenAI is configured, [`live_factcheck.py`](src/research/live_factcheck.py) builds a combined verdict from the retrieved evidence, with optional Perplexity enrichment.
12. User feedback is optionally written to the `query_feedback` table.

## Frontend Flow

The frontend is entirely Streamlit-based in [`app.py`](app.py).

The sidebar exposes two pages:

- **Analyze news**: hero area, trending trusted coverage, input text area, analysis action, prediction banner, metrics, explanation chart, related articles, optional AI brief, and yes/no feedback controls.
- **Previous searches**: the last 50 saved queries, selected-query details, and previously stored related articles.

The visual design is polished and editorial in style, but the application remains a single local process. Streamlit calls Python functions directly rather than communicating with a separate backend service.

## Database Schema

The schema is created in [`database.py`](src/storage/database.py).

### `news_queries`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER | Primary key, autoincrement |
| `query_text` | TEXT | Submitted text |
| `label` | TEXT | Prediction label |
| `confidence` | REAL | Prediction confidence |
| `created_at` | TIMESTAMP | Creation time |

### `verified_articles`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER | Primary key, autoincrement |
| `query_id` | INTEGER | Foreign key to `news_queries(id)` |
| `title` | TEXT | Article title |
| `url` | TEXT | Article URL |
| `description` | TEXT | Article description |
| `source` | TEXT | Publisher/source name |
| `image_url` | TEXT | Optional image URL |
| `similarity_score` | REAL | TF-IDF similarity score |

### `query_feedback`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER | Primary key, autoincrement |
| `query_id` | INTEGER NOT NULL | Foreign key to `news_queries(id)` |
| `feedback` | TEXT NOT NULL | User feedback value |
| `created_at` | TIMESTAMP | Creation time |

There is no user table, authentication/session table, or role and permission model. History is local machine history for the person using that SQLite database.

## External APIs and Integrations

| Integration | Status | Usage |
| --- | --- | --- |
| NewsAPI | Active | Related article retrieval in [`news_fetcher.py`](src/retrieval/news_fetcher.py) |
| OpenAI Chat Completions | Active when configured | Summary/refinement and contradiction checking in [`live_factcheck.py`](src/research/live_factcheck.py) |
| Perplexity | Optional | Additional research enrichment in [`live_factcheck.py`](src/research/live_factcheck.py) |
| Google Fact Check Tools | Implemented but not wired into the Streamlit runtime | [`factcheck_api.py`](src/retrieval/factcheck_api.py) |
| Direct scraping | Present | BBC, Reuters, and AP in [`news_scraper.py`](src/retrieval/news_scraper.py); Snopes support in [`factcheck_api.py`](src/retrieval/factcheck_api.py) |

There are no internal REST APIs exposed by the project. BeautifulSoup-based scraping code is present, but `beautifulsoup4` is not listed in [`requirements.txt`](requirements.txt).

## AI and Machine Learning Pipeline

### Runtime Pipeline

- NLTK text cleaning;
- TF-IDF vectorization;
- pickled classifier inference;
- threshold-based label mapping;
- LIME explanations; and
- TF-IDF reuse for retrieval similarity ranking.

### Dataset Artifacts

The processed data files indicate a supervised classification workflow using the schema:

- `text`;
- `label`; and
- `source_dataset`.

Available raw and processed data suggests merged sources such as ISOT, LIAR, and FakeNewsNet variants. The repository contains `combined_dataset.csv`, `train.csv`, `validation.csv`, and `test.csv`.

### Model Inventory Versus Runtime

The `models/` directory contains artifacts for logistic regression, SVM, naive Bayes, CNN-LSTM, BERT-CNN, an ensemble, and a default `fake_news_model.pkl`. However, the current Streamlit runtime loads only `models/fake_news_model.pkl` and `models/tfidf.pkl`. The UI does not expose model selection.

The JSON artifacts provide training-monitoring and model-metrics information, which supports the conclusion that multiple models were trained or intended. They do not change the active runtime path.

## Authentication and Security Scope

No authentication flow is implemented:

- no login or signup page;
- no OAuth integration;
- no JWT or application session authentication;
- no user model;
- no password storage; and
- no access-control layer.

API keys are environment-driven, but the repository does not provide production secrets management. Because the application uses local SQLite and pickled model files, deployment should be treated as a trusted single-machine setup rather than a multi-user service.

## Deployment

The supported run method is:

```bash
streamlit run app.py
```

The application loads environment variables from `.env`, uses local SQLite persistence, and reads local model artifacts. The repository does not include a Dockerfile, Docker Compose configuration, Kubernetes manifests, CI/CD configuration, reverse-proxy configuration, cloud infrastructure as code, or a FastAPI/Flask service runner.

The practical deployment model is therefore a local Streamlit process on one machine.

## Documentation and Implementation Gaps

The README describes capabilities that are absent, incomplete, or not connected to the shipped runtime:

- FastAPI backend and REST endpoints;
- user authentication and session management;
- active multi-model selection in the UI;
- production deployment scaffolding;
- several source modules described as operational components but represented mainly by decompiler artifacts or placeholders; and
- some integrations that exist in code but are not wired into the main workflow.

The README should be treated as a roadmap or aspirational architecture document until these claims are implemented and tested in the runnable application.

## Final Assessment

VeriNews is a functional academic/demo-grade fake-news verification application. Its strongest implemented components are the Streamlit UX, SQLite persistence, TF-IDF classification, LIME explanations, and evidence retrieval/ranking. Its largest risk is documentation drift: the README presents a multi-model, multi-service production platform, while the code currently behaves as a single local Streamlit application with optional external API augmentation.

For an accurate project description, deployment decision, or handoff, this report should be used as the implementation-focused reference until the README and runtime are brought into alignment.
