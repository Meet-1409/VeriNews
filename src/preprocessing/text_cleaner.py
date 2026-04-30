import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("wordnet")

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Clean and normalize raw news text for ML processing.

    Steps:
    1. Lowercasing
    2. Remove punctuation and numbers
    3. Tokenization
    4. Stopword removal
    5. Lemmatization

    Args:
        text (str): Raw input news text

    Returns:
        str: Cleaned and normalized text
    """
    if not isinstance(text, str):
        logger.warning("Non-string input received in clean_text")
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation, numbers, and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(cleaned_tokens)
