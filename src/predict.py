import json

import joblib

from src.config import MODEL_PATH, VECTORIZER_PATH, LABEL_MAP_PATH
from src.preprocessing import preprocess_text

_model = None
_vectorizer = None
_id_to_emotion = None


def _load_artifacts():
    """Load model, vectorizer, and label map into memory once."""
    global _model, _vectorizer, _id_to_emotion

    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)

        # label_map is {"anger": 0, "fear": 1, ...}
        # we need the reverse: {0: "anger", 1: "fear", ...}
        _id_to_emotion = {v: k for k, v in label_map.items()}


def predict_emotion(text: str) -> str:
    """
    Take a raw sentence and return the predicted emotion as a word,
    e.g. "joy", "sadness", etc.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty")

    _load_artifacts()

    cleaned = preprocess_text(text)
    vector = _vectorizer.transform([cleaned])
    prediction = _model.predict(vector)[0]

    return _id_to_emotion[prediction]

if __name__ == "__main__":
    print(predict_emotion("I am feeling really happy today"))
    print(predict_emotion("I am so scared right now"))