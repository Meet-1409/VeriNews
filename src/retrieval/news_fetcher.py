import logging
import requests
from typing import List, Dict
import re

from config.settings import NEWS_API_KEY, TRUSTED_SOURCES

logger = logging.getLogger(__name__)


FALLBACK_ARTICLES = [
    {
        "title": "Reuters World News",
        "description": "Live verified world coverage from Reuters.",
        "url": "https://www.reuters.com/world/",
        "source": "Reuters",
        "image_url": None,
    },
    {
        "title": "BBC News - World",
        "description": "Top international stories from BBC News.",
        "url": "https://www.bbc.com/news/world",
        "source": "BBC News",
        "image_url": None,
    },
    {
        "title": "Associated Press - Top News",
        "description": "Breaking verified headlines by AP News.",
        "url": "https://apnews.com/hub/ap-top-news",
        "source": "AP News",
        "image_url": None,
    },
]


def _compact_query(query: str) -> str:
    words = re.findall(r"[A-Za-z]{3,}", query.lower())
    stop = {
        "this", "that", "with", "from", "have", "about", "there", "their",
        "would", "could", "should", "news", "article", "headline", "likely",
        "investigation", "details", "reveals", "reveal"
    }
    filtered = [w for w in words if w not in stop]
    if not filtered:
        return "latest world news"
    return " ".join(filtered[:6])


def fetch_verified_news(
    query: str,
    page_size: int = 10
) -> List[Dict]:
    """
    Fetch verified news articles related to the given query
    from trusted news sources using NewsAPI.

    Args:
        query (str): Search query text
        page_size (int): Number of articles to retrieve

    Returns:
        List[Dict]: List of normalized news articles
    """

    if not isinstance(query, str) or query.strip() == "":
        logger.warning("Invalid query received for news retrieval")
        return []

    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY missing. Using fallback trusted links.")
        return FALLBACK_ARTICLES[:page_size]

    q = _compact_query(query)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": q,
        "sources": TRUSTED_SOURCES,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }

    try:
        logger.info("Fetching verified news from NewsAPI")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        # Retry with wider search if source-restricted query is too narrow.
        if not articles:
            params_wide = {
                "q": q,
                "language": "en",
                "pageSize": page_size,
                "sortBy": "publishedAt",
                "apiKey": NEWS_API_KEY
            }
            response = requests.get(url, params=params_wide, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])

        normalized_articles = []

        for article in articles:
            title = article.get("title") or ""
            desc = article.get("description") or ""
            if isinstance(article.get("source"), dict):
                source = article.get("source", {}).get("name") or "Unknown"
            else:
                source = "Unknown"
            normalized_articles.append({
                "title": title.strip() or "Untitled",
                "description": desc.strip() or "No description available.",
                "url": (article.get("url") or "").strip() or "#",
                "source": source.strip(),
                "image_url": article.get("urlToImage") or None,
            })

        if normalized_articles:
            logger.info(f"Retrieved {len(normalized_articles)} articles")
            return normalized_articles
        logger.info("No articles from API; using fallback trusted links")
        return FALLBACK_ARTICLES[:page_size]

    except requests.exceptions.RequestException as e:
        logger.error(f"News API request failed: {e}")
        return FALLBACK_ARTICLES[:page_size]
