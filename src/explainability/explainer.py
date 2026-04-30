import logging
import joblib
import numpy as np
from pathlib import Path

from lime.lime_text import LimeTextExplainer
from src.preprocessing.text_cleaner import clean_text

logger = logging.getLogger(__name__)


class FakeNewsExplainer:
    """
    Explainable AI module for fake news classification
    using LIME.
    """

    def __init__(
        self,
        model_path: str = "models/fake_news_model.pkl",
        vectorizer_path: str = "models/tfidf.pkl"
    ):
        logger.info("Loading model and vectorizer for explainability")

        # Resolve paths relative to project root if they're relative paths
        if not Path(model_path).is_absolute():
            # Try to find project root (parent of src directory)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            model_path = project_root / model_path
            vectorizer_path = project_root / vectorizer_path

        self.model = joblib.load(str(model_path))
        self.vectorizer = joblib.load(str(vectorizer_path))

        self.class_names = ["Fake", "Real"]
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def _predict_proba_wrapper(self, texts):
        """
        Wrapper to provide probability-like output for LIME.
        """
        cleaned_texts = [clean_text(t) for t in texts]
        vectors = self.vectorizer.transform(cleaned_texts)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(vectors)

        # Fallback for models like LinearSVC
        preds = self.model.predict(vectors)
        proba = np.zeros((len(preds), 2))
        for i, p in enumerate(preds):
            proba[i][p] = 1.0
        return proba

    def explain(self, text: str, num_features: int = 10) -> list:
        """
        Generate explanation for a single news article.

        Args:
            text (str): Raw news text
            num_features (int): Number of words to explain

        Returns:
            list: List of (word, weight) tuples
        """

        if not isinstance(text, str) or text.strip() == "":
            logger.warning("Invalid input received for explanation")
            return []

        explanation = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self._predict_proba_wrapper,
            num_features=num_features
        )

        return explanation.as_list()
