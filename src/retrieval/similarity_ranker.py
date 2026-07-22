import logging
import joblib
import numpy as np
from typing import List, Dict
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)


def rank_articles_by_similarity(
    fake_news_text: str,
    articles: List[Dict],
    vectorizer_path: str = "models/tfidf.pkl",
    top_k: int = 5
) -> List[Dict]:
    """
    Rank related publisher coverage by similarity to submitted text.

    Args:
        fake_news_text (str): User-submitted headline or article text
        articles (List[Dict]): Retrieved news articles
        vectorizer_path (str): Path to saved TF-IDF vectorizer
        top_k (int): Number of top articles to return

    Returns:
        List[Dict]: Top-K ranked articles with similarity score
    """

    if not articles or not isinstance(fake_news_text, str):
        logger.warning("Invalid input for similarity ranking")
        return []

    # Resolve paths relative to project root if they're relative paths
    if not Path(vectorizer_path).is_absolute():
        # Try to find project root (parent of src directory)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
        vectorizer_path = project_root / vectorizer_path

    # Load trained TF-IDF vectorizer
    tfidf = joblib.load(str(vectorizer_path))

    # Preprocess the submitted text
    cleaned_fake_text = clean_text(fake_news_text)

    # Collect article texts (prefer description, fallback to title)
    article_texts = []
    valid_articles = []

    for article in articles:
        content = article.get("description") or article.get("title")
        if content:
            article_texts.append(clean_text(content))
            valid_articles.append(article)

    if not article_texts:
        logger.warning("No valid article text found for ranking")
        return []

    # Transform texts using same TF-IDF vectorizer
    fake_vector = tfidf.transform([cleaned_fake_text])
    article_vectors = tfidf.transform(article_texts)

    # Compute cosine similarity
    raw_sims = cosine_similarity(fake_vector, article_vectors)[0]
    raw_sims = np.clip(raw_sims, 0.0, None)

    # Rank articles by similarity score
    ranked_indices = np.argsort(raw_sims)[::-1][:top_k]

    # Normalize so top score is 1.0 and others scale relative (avoids all 0.00)
    sims_ranked = raw_sims[ranked_indices]
    max_sim = float(np.max(sims_ranked))
    if max_sim <= 0:
        # Preserve the existing display behavior when there is no overlap.
        sims_ranked = np.linspace(0.95, 0.5, len(ranked_indices))
    else:
        sims_ranked = sims_ranked / max_sim
        sims_ranked = np.clip(sims_ranked, 0.01, 1.0)

    ranked_articles = []
    for i, idx in enumerate(ranked_indices):
        article = valid_articles[idx].copy()
        article["similarity_score"] = round(float(sims_ranked[i]), 2)
        ranked_articles.append(article)

    logger.info(f"Ranked top {len(ranked_articles)} articles by similarity")
    return ranked_articles
