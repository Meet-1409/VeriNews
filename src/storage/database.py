"""
VeriNews SQLite persistence layer.

Stores queries, verified articles and user feedback.
Schema is forward-compatible: ALTER TABLE migrations run on startup.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "verinews.db"


class VeriNewsDB:
    """SQLite-based persistence for VeriNews."""

    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or DB_PATH)
        self._initialize_database()

    # ── connection ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ── schema ───────────────────────────────────────────────────────────

    def _initialize_database(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_queries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text  TEXT    NOT NULL,
                    label       TEXT    NOT NULL,
                    confidence  REAL,
                    prob_fake   REAL,
                    prob_real   REAL,
                    model_key   TEXT    DEFAULT 'ensemble',
                    created_at  TEXT    NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verified_articles (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id         INTEGER NOT NULL,
                    title            TEXT,
                    url              TEXT,
                    description      TEXT,
                    source           TEXT,
                    image_url        TEXT,
                    similarity_score REAL,
                    FOREIGN KEY(query_id) REFERENCES news_queries(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_feedback (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id   INTEGER NOT NULL,
                    feedback   TEXT    NOT NULL CHECK(feedback IN ('yes','no')),
                    created_at TEXT    NOT NULL,
                    FOREIGN KEY(query_id) REFERENCES news_queries(id)
                )
            """)
            conn.commit()
            # Migrate older schemas that might be missing columns
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns that were introduced after initial release."""
        existing: set[str] = {
            row[1] for row in conn.execute("PRAGMA table_info(news_queries)")
        }
        new_cols = [
            ("prob_fake",  "REAL"),
            ("prob_real",  "REAL"),
            ("model_key",  "TEXT DEFAULT 'ensemble'"),
        ]
        for col, typedef in new_cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE news_queries ADD COLUMN {col} {typedef}")
        conn.commit()

    # ── writes ───────────────────────────────────────────────────────────

    def save_query(
        self,
        query_text: str,
        label: str,
        confidence: float,
        prob_fake: float = 0.0,
        prob_real: float = 0.0,
        model_key: str  = "ensemble",
    ) -> int:
        """Insert a new query and return its row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO news_queries
                    (query_text, label, confidence, prob_fake, prob_real, model_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (query_text, label, confidence, prob_fake, prob_real,
                 model_key, datetime.utcnow().isoformat()),
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def save_verified_articles(self, query_id: int, articles: list[dict]) -> None:
        """Bulk-insert verified articles for a query."""
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO verified_articles
                    (query_id, title, url, description, source, image_url, similarity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        query_id,
                        a.get("title"),
                        a.get("url"),
                        a.get("description"),
                        a.get("source"),
                        a.get("image_url"),
                        a.get("similarity_score"),
                    )
                    for a in articles
                ],
            )
            conn.commit()

    def save_feedback(self, query_id: int, feedback: str) -> None:
        """Store 'yes' or 'no' feedback for a query."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_feedback (query_id, feedback, created_at)
                VALUES (?, ?, ?)
                """,
                (query_id, feedback, datetime.utcnow().isoformat()),
            )
            conn.commit()

    # ── reads ────────────────────────────────────────────────────────────

    def get_query_by_id(self, query_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM news_queries WHERE id = ?", (query_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_recent_queries(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query_text, label, confidence, prob_fake,
                       prob_real, model_key, created_at
                FROM news_queries
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_verified_articles_for_query(self, query_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT title, url, description, source, similarity_score
                FROM verified_articles
                WHERE query_id = ?
                ORDER BY similarity_score DESC
                """,
                (query_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_count(self, query_id: int | None = None) -> dict[str, int]:
        """
        Return yes/no counts. If query_id is given, counts for that query only;
        otherwise global counts for the analytics view.
        """
        with self._connect() as conn:
            if query_id is not None:
                rows = conn.execute(
                    "SELECT feedback, COUNT(*) FROM query_feedback "
                    "WHERE query_id = ? GROUP BY feedback",
                    (query_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT feedback, COUNT(*) FROM query_feedback GROUP BY feedback"
                ).fetchall()
        result: dict[str, int] = {"yes": 0, "no": 0}
        for feedback, count in rows:
            if feedback in result:
                result[feedback] = count
        return result

    def fetch_similar_past_query(self, query_text: str) -> dict:
        """Return the most recent identical query if it exists (simple caching)."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, label, confidence
                FROM news_queries
                WHERE query_text = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (query_text,),
            ).fetchone()
        return dict(row) if row else {}
