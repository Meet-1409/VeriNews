"""
Web scraper to collect news articles from reputable sources for custom dataset.
This scraper focuses on trusted sources: AP News, Reuters, BBC, etc.
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NewsArticleScraper:
    """Scrapes news articles from reputable sources for ground truth data."""

    def __init__(self):
        """Initialize scraper with news source endpoints."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # These sources are known to be reliable news outlets
        self.trusted_sources = {
            'bbc': 'https://www.bbc.com/news',
            'reuters': 'https://www.reuters.com',
            'ap_news': 'https://apnews.com',
            'bbc_world': 'https://www.bbc.com/news/world'
        }

    def scrape_bbc_news(self, limit: int = 50) -> List[Dict]:
        """
        Scrape BBC News headlines (reputable, fact-checked source).
        
        Args:
            limit: Number of articles to scrape
            
        Returns:
            List of article dictionaries with text and metadata
        """
        articles = []
        try:
            response = requests.get(self.trusted_sources['bbc'], headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # BBC news items
            h2_tags = soup.find_all('h2')
            for h2 in h2_tags[:limit]:
                text = h2.get_text(strip=True)
                if text and len(text) > 20:  # Filter out very short titles
                    articles.append({
                        'text': text,
                        'source': 'BBC News',
                        'label': 1,  # BBC is reputable (real news)
                        'confidence': 0.95,
                        'timestamp': datetime.now().isoformat()
                    })

            logger.info(f"Scraped {len(articles)} articles from BBC News")
            return articles

        except Exception as e:
            logger.error(f"Error scraping BBC: {str(e)}")
            return []

    def scrape_ap_news(self, limit: int = 50) -> List[Dict]:
        """
        Scrape AP News headlines (fact-checked, reputable).
        
        Args:
            limit: Number of articles to scrape
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            response = requests.get(self.trusted_sources['ap_news'], headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article links and titles
            article_links = soup.find_all('a', class_='Component-headline')
            for link in article_links[:limit]:
                text = link.get_text(strip=True)
                if text and len(text) > 20:
                    articles.append({
                        'text': text,
                        'source': 'AP News',
                        'label': 1,  # AP is reputable (real news)
                        'confidence': 0.95,
                        'timestamp': datetime.now().isoformat()
                    })

            logger.info(f"Scraped {len(articles)} articles from AP News")
            return articles

        except Exception as e:
            logger.error(f"Error scraping AP News: {str(e)}")
            return []

    def scrape_reuters(self, limit: int = 50) -> List[Dict]:
        """
        Scrape Reuters headlines (fact-checked, reputable).
        
        Args:
            limit: Number of articles to scrape
            
        Returns:
            List of article dictionaries
        """
        articles = []
        try:
            response = requests.get(self.trusted_sources['reuters'], headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find headlines
            h3_tags = soup.find_all('h3')
            for h3 in h3_tags[:limit]:
                text = h3.get_text(strip=True)
                if text and len(text) > 20:
                    articles.append({
                        'text': text,
                        'source': 'Reuters',
                        'label': 1,  # Reuters is reputable (real news)
                        'confidence': 0.95,
                        'timestamp': datetime.now().isoformat()
                    })

            logger.info(f"Scraped {len(articles)} articles from Reuters")
            return articles

        except Exception as e:
            logger.error(f"Error scraping Reuters: {str(e)}")
            return []

    def collect_all_trusted_news(self, articles_per_source: int = 30) -> List[Dict]:
        """
        Collect verified real news from multiple trusted sources.
        
        Args:
            articles_per_source: Number of articles to scrape from each source
            
        Returns:
            Combined list of verified real news articles
        """
        all_articles = []

        logger.info("Starting collection of verified real news...")
        all_articles.extend(self.scrape_bbc_news(articles_per_source))
        all_articles.extend(self.scrape_reuters(articles_per_source))
        all_articles.extend(self.scrape_ap_news(articles_per_source))

        logger.info(f"Total verified articles collected: {len(all_articles)}")
        return all_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = NewsArticleScraper()
    articles = scraper.collect_all_trusted_news(articles_per_source=20)
    
    # Save to file for inspection
    with open('data/raw/scraped_real_news.json', 'w') as f:
        json.dump(articles, f, indent=2)
    
    print(f"Saved {len(articles)} verified articles to scraped_real_news.json")
