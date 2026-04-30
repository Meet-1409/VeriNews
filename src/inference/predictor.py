import logging
import joblib
import numpy as np
from pathlib import Path

from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class FakeNewsPredictor:
    """
    Inference pipeline for fake news detection.
    Loads trained model and TF-IDF vectorizer and
    provides prediction on unseen news text.
    """

    def __init__(
        self,
        model_path: str = "models/fake_news_model.pkl",
        vectorizer_path: str = "models/tfidf.pkl"
    ):
        logger.info("Loading trained model and TF-IDF vectorizer")

        # Resolve paths relative to project root if they're relative paths
        if not Path(model_path).is_absolute():
            # Try to find project root (parent of src directory)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            model_path = project_root / model_path
            vectorizer_path = project_root / vectorizer_path

        self.model = joblib.load(str(model_path))
        self.vectorizer = joblib.load(str(vectorizer_path))

    def predict(self, text: str) -> dict:
        """
        Predict whether the given news text is Fake or Real.

        Args:
            text (str): Raw news article text

        Returns:
            dict: Prediction result containing label and confidence
        """

        if not isinstance(text, str) or text.strip() == "":
            logger.warning("Empty or invalid input received for prediction")
            return {
                "label": "Invalid",
                "confidence": 0.0
            }

        # Preprocess text
        cleaned_text = clean_text(text)

        # Transform text using saved TF-IDF vectorizer
        text_vector = self.vectorizer.transform([cleaned_text])

        # Default values
        label = "Fake"
        confidence = 1.0

        # Use calibrated probabilities when available to make a softer decision.
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(text_vector)[0]
            classes = list(getattr(self.model, "classes_", [0, 1]))

            # Find probability of Fake (0) and Real (1)
            try:
                idx_fake = classes.index(0)
                idx_real = classes.index(1)
            except ValueError:
                # Fallback to first/second column
                idx_fake, idx_real = 0, 1

            prob_fake = float(probs[idx_fake])
            prob_real = float(probs[idx_real])

            # Strong bias toward Real/Uncertain for real-world news; Fake only when very confident.
            if prob_fake >= 0.90:
                label = "Fake"
                confidence = prob_fake
            elif prob_real >= 0.22:
                label = "Real"
                confidence = prob_real
            else:
                label = "Uncertain"
                confidence = max(prob_fake, prob_real)
        else:
            # Models without predict_proba – keep original behaviour
            prediction = self.model.predict(text_vector)[0]
            label = "Real" if prediction == 1 else "Fake"
            confidence = 1.0  # treated as high-confidence decision

        return {
            "label": label,
            "confidence": round(float(confidence), 3)
        }
