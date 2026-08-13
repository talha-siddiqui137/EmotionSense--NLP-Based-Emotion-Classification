import json

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import TEST_PATH, MODEL_PATH, VECTORIZER_PATH, LABEL_MAP_PATH
from src.data_loader import load_dataset, encode_labels
from src.preprocessing import preprocess_text


def evaluate_model():
    # 1. Load saved artifacts
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    # 2. Load and prepare test data
    test_df = load_dataset(TEST_PATH)
    test_df = encode_labels(test_df, label_map)
    test_df["text"] = test_df["text"].apply(preprocess_text)

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["emotion"]

    # 3. Predict and score
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()