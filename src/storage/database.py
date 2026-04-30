import sqlite3
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)


class VeriNewsDB:
    """
    SQLite-based persistence layer for VeriNews.
    Stores past queries and verified articles.
    """

    def __init__(self, db_path: str = "verinews.db"):
        self.db_path = db_path
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        logger.info("Initializing SQLite database")

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT,
                    label TEXT,
                    confidence REAL,
                    created_at TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verified_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    title TEXT,
                    url TEXT,
                    description TEXT,
                    source TEXT,
                    image_url TEXT,
                    similarity_score REAL,
                    FOREIGN KEY(query_id) REFERENCES news_queries(id)
                )
            """)

            cursor.execute("PRAGMA table_info(verified_articles)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            if "description" not in existing_columns:
                cursor.execute("ALTER TABLE verified_articles ADD COLUMN description TEXT")
            if "image_url" not in existing_columns:
                cursor.execute("ALTER TABLE verified_articles ADD COLUMN image_url TEXT")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    feedback TEXT NOT NULL,
                    created_at TIMESTAMP,
                    FOREIGN KEY(query_id) REFERENCES news_queries(id)
                )
            """)

            conn.commit()

    def save_query(self, query_text: str, label: str, confidence: float) -> int:
        logger.info("Saving user query to database")

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO news_queries (query_text, label, confidence, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (query_text, label, confidence, datetime.now())
            )
            conn.commit()
            return cursor.lastrowid

    def save_verified_articles(
        self,
        query_id: int,
        articles: List[Dict]
    ):
        logger.info("Saving verified articles to database")

        with self._connect() as conn:
            cursor = conn.cursor()
            for article in articles:
                cursor.execute(
                    """
                    INSERT INTO verified_articles
                    (query_id, title, url, description, source, image_url, similarity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        article.get("title"),
                        article.get("url"),
                        article.get("description"),
                        article.get("source"),
                        article.get("image_url"),
                        article.get("similarity_score")
                    )
                )
            conn.commit()

    def fetch_similar_past_query(self, query_text: str) -> Dict:
        """
        Fetch the most recent identical query if it exists.
        (Simple memory reuse – safe for academics)
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, label, confidence
                FROM news_queries
                WHERE query_text = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (query_text,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    "query_id": row[0],
                    "label": row[1],
                    "confidence": row[2]
                }
            return {}

    def get_recent_queries(self, limit: int = 50) -> List[Dict]:
        """
        Return recent queries for history view.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, query_text, label, confidence, created_at
                FROM news_queries
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        results: List[Dict] = []
        for r in rows:
            results.append(
                {
                    "id": r[0],
                    "query_text": r[1],
                    "label": r[2],
                    "confidence": r[3],
                    "created_at": r[4],
                }
            )
        return results

    def get_verified_articles_for_query(self, query_id: int) -> List[Dict]:
        """
        Return verified articles stored for a given query.
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT title, url, description, source, image_url, similarity_score
                FROM verified_articles
                WHERE query_id = ?
                ORDER BY similarity_score DESC
                """,
                (query_id,),
            )
            rows = cursor.fetchall()

        articles: List[Dict] = []
        for r in rows:
            articles.append(
                {
                    "title": r[0],
                    "url": r[1],
                    "description": r[2],
                    "source": r[3],
                    "image_url": r[4],
                    "similarity_score": r[5],
                }
            )
        return articles

    def save_feedback(self, query_id: int, feedback: str) -> None:
        """Store user feedback (yes/no) for a verification. Used for improvement and analytics."""
        if feedback not in ("yes", "no"):
            return
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_feedback (query_id, feedback, created_at)
                VALUES (?, ?, ?)
                """,
                (query_id, feedback, datetime.now()),
            )
            conn.commit()

    def get_feedback_count(self) -> Dict[str, int]:
        """Return counts of yes/no feedback for display or analytics."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT feedback, COUNT(*) FROM query_feedback GROUP BY feedback"
            )
            rows = cursor.fetchall()
        result = {"yes": 0, "no": 0}
        for row in rows:
            if row[0] in result:
                result[row[0]] = row[1]
        return result
