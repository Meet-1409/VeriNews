# API keys, constants
import os
from pathlib import Path

# Load .env from project root if present (keeps keys out of code)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# News API Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Trusted news sources (comma-separated string for NewsAPI)
# Example: "bbc-news,reuters,associated-press,the-guardian-uk"
TRUSTED_SOURCES = os.getenv("TRUSTED_SOURCES", "bbc-news,reuters,associated-press")

# Live fact-check APIs
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Optional key for future Cursor-connected integrations
CURSOR_API_KEY = os.getenv("CURSOR_API_KEY", "")
