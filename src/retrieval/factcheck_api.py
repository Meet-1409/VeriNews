"""
Integrate with fact-checking APIs (Google FactCheck API, Snopes) to auto-label claims.
This provides ground truth labels for custom dataset creation.
"""

import requests
import json
import logging
from typing import List, Dict, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class FactCheckIntegration:
    """Integrates with fact-checking APIs to auto-label news claims."""

    def __init__(self, google_api_key: str = None):
        """
        Initialize fact-check integration.
        
        Args:
            google_api_key: Google FactCheck API key (optional for enhanced capability)
        """
        self.google_api_key = google_api_key or os.getenv('GOOGLE_FACTCHECK_API_KEY', '')
        self.google_api_url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
        self.snopes_api_url = 'https://www.snopes.com'

    def query_google_factcheck(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Query Google FactCheck API for fact-checked claims.
        
        Args:
            query: Search query (news topic, claim, etc.)
            limit: Maximum results to return
            
        Returns:
            List of fact-checked claims with labels
        """
        if not self.google_api_key:
            logger.warning("Google FactCheck API key not configured. Skipping Google FactCheck.")
            return []

        try:
            params = {
                'query': query,
                'key': self.google_api_key,
                'languageCode': 'en',
                'maxAgeDays': 30  # Recent fact-checks only
            }

            response = requests.get(self.google_api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for claim in data.get('claims', [])[:limit]:
                # Parse claim ratings
                rating = claim.get('claimReview', [{}])[0].get('textualRating', 'UNKNOWN')
                
                # Map rating to binary label
                # True ratings: TRUE, MOSTLY_TRUE, CORRECT
                # False ratings: FALSE, MOSTLY_FALSE, INCORRECT
                label = 1 if rating in ['TRUE', 'MOSTLY_TRUE', 'CORRECT'] else 0

                results.append({
                    'text': claim.get('text', ''),
                    'claim_date': claim.get('claimDate', ''),
                    'source': claim.get('claimant', 'Unknown'),
                    'rating': rating,
                    'label': label,
                    'confidence': 0.9,  # High confidence from fact-check API
                    'api_source': 'Google FactCheck',
                    'timestamp': datetime.now().isoformat()
                })

            logger.info(f"Retrieved {len(results)} fact-checked claims from Google API for query: {query}")
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Google FactCheck API: {str(e)}")
            return []
        except KeyError as e:
            logger.error(f"Unexpected API response format: {str(e)}")
            return []

    def get_snopes_fact_checks(self, limit: int = 20) -> List[Dict]:
        """
        Get recent fact-checks from Snopes (web scraping approach).
        
        Args:
            limit: Number of recent articles to retrieve
            
        Returns:
            List of fact-checked claims from Snopes
        """
        try:
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # Snopes recent articles
            response = requests.get('https://www.snopes.com', headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            results = []
            article_links = soup.find_all('a', class_='article-link')[:limit]

            for link in article_links:
                title = link.get_text(strip=True)
                url = link.get('href', '')

                # Snopes articles are fact-checked (considered real/verified)
                results.append({
                    'text': title,
                    'source': 'Snopes',
                    'url': url,
                    'label': 1,  # Snopes fact-checks = verified/debunked (real analysis)
                    'confidence': 0.9,
                    'api_source': 'Snopes',
                    'timestamp': datetime.now().isoformat()
                })

            logger.info(f"Retrieved {len(results)} recent fact-checks from Snopes")
            return results

        except Exception as e:
            logger.error(f"Error retrieving Snopes data: {str(e)}")
            return []

    def combine_factcheck_sources(self) -> List[Dict]:
        """
        Combine fact-checked claims from multiple sources.
        This creates a labeled dataset of verified claims.
        
        Returns:
            Combined list of fact-checked items with reliable labels
        """
        all_claims = []

        logger.info("Combining fact-check sources...")

        # Get from Google FactCheck for multiple recent topics
        topics = ['election', 'health', 'politics', 'technology', 'environment']
        for topic in topics:
            all_claims.extend(self.query_google_factcheck(topic, limit=5))

        # Get from Snopes
        all_claims.extend(self.get_snopes_fact_checks(limit=20))

        logger.info(f"Combined total: {len(all_claims)} fact-checked claims")
        return all_claims

    def label_with_confidence(self, text: str) -> Tuple[int, float]:
        """
        Label a piece of text based on fact-checking, with confidence score.
        
        Args:
            text: News text to label
            
        Returns:
            Tuple of (label: 0=fake/1=real, confidence: 0.0-1.0)
        """
        # For now, requires external input
        # In production, would integrate with live fact-check APIs
        logger.warning("Direct text labeling requires implementation of real-time API calls")
        return None, 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fact_check = FactCheckIntegration()
    
    # Collect all fact-checked claims
    all_claims = fact_check.combine_factcheck_sources()
    
    # Save to file
    with open('data/raw/factchecked_claims.json', 'w') as f:
        json.dump(all_claims, f, indent=2)
    
    print(f"Saved {len(all_claims)} fact-checked claims")
