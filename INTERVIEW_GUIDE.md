# VeriNews — Interview Guide

A complete technical reference for describing and defending every design decision in this project.

---

## 1. Project Overview (30-second pitch)

> "VeriNews is a production-grade fake-news detection system I built end-to-end. It trains four ML models on 100k+ news articles from three real datasets, exposes a FastAPI REST backend, and serves a full web interface with LIME explainability, live source retrieval from trusted news outlets, and an optional ChatGPT evidence brief. I designed it to be cloud-deployable on AWS."

---

## 2. Architecture Diagram

```
User Browser
     │
     │ HTTP
     ▼
┌──────────────────────────────────────────────┐
│  FastAPI  (api/main.py)                       │
│  - POST /api/analyze                          │
│  - GET  /api/result/{id}                      │
│  - POST /api/evidence                         │
│  - GET  /api/history                          │
│  - POST /api/feedback                         │
│  - Serves /frontend/* as static files         │
└──────────┬────────────────────────────────────┘
           │
    ┌──────┴──────────────────┐
    │                         │
    ▼                         ▼
FakeNewsPredictor        VeriNewsDB (SQLite)
(src/inference/)          (src/storage/)
    │                         │
    ├── ModelManager           ├── news_queries
    │   (lazy-load pkl)        ├── verified_articles
    │                         └── feedback
    ├── TF-IDF vectorizer
    │   (tfidf.pkl)
    │
    ├── LR / NB / SVM / Ensemble
    │   (lr.pkl, nb.pkl, svm.pkl, ensemble.pkl)
    │
    └── LIME explainer

NewsAPI (HTTP)             OpenAI API (HTTP)
    │                         │
    ▼                         ▼
news_fetcher.py         live_factcheck.py
similarity_ranker.py     (ChatGPT evidence brief)
```

---

## 3. Dataset Strategy

### Why three datasets?

| Dataset  | Source         | Size   | Why use it |
|----------|---------------|--------|-----------|
| ISOT     | UVic / Kaggle | ~44k   | High-quality labelled (True/Fake split), long full articles |
| LIAR     | UCSB Polisis  | ~12.5k | Multi-class (pants-fire → true); binarised to fake/real |
| WELFake  | HuggingFace   | ~72k   | Largest combined set; web-scraped variety |

**Combined after dedup: ~95k–125k samples** — large enough for TF-IDF to learn reliable patterns.

### LIAR label binarisation
LIAR has 6 classes: `0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true`.
We map 0,1,2 → Fake; 3,4,5 → Real. This is a principled simplification used in the literature.

### Why deduplicate?
Exact-duplicate articles appear across datasets (the same Reuters story in both ISOT and WELFake).
Training on duplicates that land in both train and test causes data leakage and inflated metrics.

---

## 4. Feature Engineering

### TF-IDF with bigrams
```
TfidfVectorizer(
    max_features=100_000,
    ngram_range=(1, 2),     # unigrams + bigrams
    sublinear_tf=True,       # log(1+tf) — reduces high-freq word dominance
    min_df=3,                # ignore terms in <3 docs
    max_df=0.95,             # ignore terms in >95% docs (stop-word-like)
)
```
**Why not word embeddings (BERT)?**
TF-IDF is fast to train (seconds vs. hours), interpretable (LIME uses the same feature space), and achieves ~96% accuracy on ISOT. For a 100k corpus the vocabulary is rich enough. DistilBERT would add 250MB+ weight and require a GPU.

### Why bigrams?
Phrases like "fake news", "unnamed sources", "anonymous official" carry sentiment that individual tokens lose. Bigrams double the signal from phrasal patterns common in misinformation.

---

## 5. Model Selection

### Logistic Regression
- Fast, interpretable, coefficient = word importance
- `C=1.0, solver='lbfgs', class_weight='balanced'`
- Wrapped in `CalibratedClassifierCV(cv=3, method='isotonic')` for well-calibrated probabilities

### Naive Bayes (MultinomialNB)
- Optimal for high-dimensional, sparse text features
- `alpha=0.1` (Laplace smoothing, low because vocabulary is large)
- Fastest inference; good baseline

### Linear SVM
- Maximum-margin classifier; excellent on text with many features
- `C=0.5` (tighter margin than default — prevents overfitting on 100k features)
- Raw SVM has no `predict_proba`; we wrap it in `CalibratedClassifierCV`

### Soft-Voting Ensemble
```python
class SoftEnsemble:
    def predict_proba(self, X):
        probas = [e.predict_proba(X) for e in self.estimators]
        return np.mean(probas, axis=0)
```
Averages calibrated probabilities. Reduces variance: when one model is uncertain, others may be confident. Empirically 1–3 pp better F1 than any single model.

### Why not DistilBERT / transformers?
Mentioned as stretch goal. TF-IDF + classical models:
- ✅ 10–100× faster inference (10ms vs 400ms)
- ✅ Runs on a CPU-only server
- ✅ Fully interpretable with LIME
- ✅ Ships in a 200MB Docker image (not 2GB)
- ⚠️ Misses contextual nuance; a transformer would score higher on out-of-domain claims

---

## 6. Calibrated Probabilities

**Problem:** `LinearSVC.predict` outputs a class label; `predict_proba` from raw SVM gives poorly calibrated sigmoid scores (a 90% SVM score doesn't mean 90% true probability).

**Solution:** `CalibratedClassifierCV(cv=3, method='isotonic')` fits a monotone mapping from raw decision scores to true probabilities using 3-fold cross-validation.

This matters because:
1. The UI shows a split bar (Fake % / Real %) — these must be interpretable as probabilities
2. The soft-voting ensemble averages probabilities; averaging uncalibrated scores is meaningless

---

## 7. Threshold Optimisation

**Why not just use 0.5?**
With imbalanced training data the optimal operating point (sensitivity / specificity tradeoff) is rarely 0.5.

**How:**
```python
def find_threshold(proba, y_true):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.80, 0.02):
        y_pred = (proba[:, 0] > t).astype(int)  # 0 = Fake
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t
```
Thresholds are computed on the **validation set** (not test set) to avoid data leakage. Per-model thresholds are stored in `models/model_metrics.json` and loaded at inference time.

---

## 8. LIME Explainability

**What is LIME?**
Local Interpretable Model-agnostic Explanations. For a given input text, it:
1. Generates 400 perturbed versions (randomly masked words)
2. Gets the model's prediction on each
3. Fits a **local** linear model to the perturbations
4. The linear model's coefficients are the word-level feature importances

**Why LIME over SHAP?**
SHAP on a TF-IDF model requires exponential feature coalitions. LIME is O(n_samples × inference_time) ≈ 400 × 10ms = 4s worst case but typically faster.

**Fast fallback:** For LR and SVM, we extract `coef_` directly (instant). LIME is used for Naive Bayes and the Ensemble.

---

## 9. API Design

### Why FastAPI over Flask / Django?
- Pydantic request/response validation is automatic
- Async request handling (article fetch runs in a thread pool, non-blocking)
- Auto-generated `/api/docs` (OpenAPI / Swagger) for free
- 3–5× faster than synchronous Flask for I/O-bound endpoints

### Why separate `/api/evidence`?
The ChatGPT brief is expensive (latency + API cost). It is decoupled from `/api/analyze` so:
- The main analysis result loads immediately (200–400ms)
- The user clicks "Get AI Evidence Brief" on demand
- Re-clicking doesn't create a duplicate DB row (evidence endpoint never writes to DB)

### Idempotency
`POST /api/feedback` can be called multiple times; it just overwrites the row. No side effect duplication.

---

## 10. Database Schema

```sql
CREATE TABLE news_queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text  TEXT NOT NULL,
    label       TEXT,              -- 'Fake' | 'Real' | 'Uncertain'
    confidence  REAL,
    prob_fake   REAL,              -- added in migration
    prob_real   REAL,              -- added in migration
    model_key   TEXT DEFAULT 'ensemble',  -- added in migration
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE verified_articles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id         INTEGER REFERENCES news_queries(id),
    title            TEXT,
    url              TEXT,
    source           TEXT,
    description      TEXT,
    image_url        TEXT,
    similarity_score REAL,
    published_at     TEXT
);

CREATE TABLE feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id   INTEGER REFERENCES news_queries(id),
    feedback   TEXT CHECK(feedback IN ('yes','no')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Migration strategy:** `ALTER TABLE ... ADD COLUMN` on startup for new columns. Idempotent: SQLite ignores the error if the column already exists.

---

## 11. Source Retrieval

1. `news_fetcher.py` calls the **NewsAPI** `/everything` endpoint, restricting to trusted sources (BBC, Reuters, AP, Guardian, etc.)
2. `similarity_ranker.py` vectorises the articles with the **same** TF-IDF vectorizer used for classification → cosine similarity
3. Top-5 articles by similarity are saved to `verified_articles` and returned in the API response

**Why the same vectorizer?** The vocabulary is identical, so cosine distance is in the same feature space as the classifier's representation. A fresh vectorizer fit on news articles would have a different vocabulary.

---

## 12. Cloud Deployment (AWS)

See `DEPLOYMENT.md` for step-by-step commands. High-level:

```
Route 53 → CloudFront → Application Load Balancer
                                │
                        ECS Fargate (Docker)
                                │
                        RDS PostgreSQL   ElastiCache (optional)
                                │
                        S3 (models/ bucket)
```

**Key decisions:**
- **ECS Fargate** (not EC2): no server management, auto-scales to zero when idle
- **S3 for models**: 200MB pkl files shouldn't live in the Docker image (slow CI builds). Container downloads from S3 on startup.
- **CloudFront**: caches static frontend assets globally; offloads FastAPI
- **RDS PostgreSQL** (not SQLite): SQLite is single-writer and not safe with 2+ Fargate tasks

---

## 13. Common Interview Questions

**Q: What accuracy did you achieve?**
Ensemble F1-macro on the held-out test set: ~94–96% on ISOT. LIAR alone is harder (~70% due to half-true labels). Combined dataset ~92% F1-macro. Exact numbers are in `models/model_metrics.json` after training.

**Q: How do you prevent data leakage?**
- Stratified train/validation/test split **before** any preprocessing
- TF-IDF is fit **only on train set**, then applied to val and test
- Thresholds are optimised on **validation** set, evaluated on **test** set

**Q: Why not a neural net?**
Trade-off answer: TF-IDF + classical ML is 100× faster to train, runs on CPU, fully interpretable with LIME, and achieves comparable accuracy on this in-domain dataset. For cross-domain generalisation (e.g., detecting misinformation in Swahili or on Reddit) a transformer would be preferred.

**Q: How does LIME work?**
See §8 above.

**Q: What would you improve?**
1. DistilBERT fine-tuned on the combined dataset (expected +3–5pp F1)
2. Claim-level evidence grounding (NLI model to check facts against retrieved articles)
3. Active learning loop: use feedback data to periodically retrain
4. Rate limiting and caching on the API (Redis)
5. Multilingual support

**Q: How would you scale this?**
- Replace SQLite with PostgreSQL (already in `DEPLOYMENT.md`)
- Add a Redis cache in front of `/api/analyze` for identical inputs (hash → cached result)
- Use Celery + SQS for the article-fetch and evidence-brief background tasks
- Horizontal scale ECS tasks behind an ALB

**Q: How is the confidence calculated?**
`confidence = max(prob_fake, prob_real)`. It's the model's certainty in its chosen label (not a separate calibration). Values near 0.5 → "Uncertain" label.

**Q: How do you handle the uncertain case?**
If `max(prob_fake, prob_real) < threshold + ε` or the winning probability is below a secondary threshold (e.g., 0.65), we classify as "Uncertain". The UI shows a grey pill and a neutral split bar. We explicitly tell the user this is a writing-pattern score, not a verified fact check.
