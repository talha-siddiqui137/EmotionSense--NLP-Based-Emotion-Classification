import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

from src.config import TRAIN_PATH, TEST_PATH, MODEL_PATH, VECTORIZER_PATH, LABEL_MAP_PATH, MODELS_DIR
from src.data_loader import load_dataset, build_label_map, encode_labels
from src.preprocessing import preprocess_text


def train_model():
    # 1. Load raw data
    train_df = load_dataset(TRAIN_PATH)

    # 2. Build label map from training data only, then apply it
    label_map = build_label_map(train_df)
    train_df = encode_labels(train_df, label_map)

    # 3. Clean the text (same function used later at prediction time)
    train_df["text"] = train_df["text"].apply(preprocess_text)

    # 4. Turn cleaned text into TF-IDF numbers
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["emotion"]

    # 5. Train the model
    model = LinearSVC(C=100, max_iter=5000)
    model.fit(X_train, y_train)

    # 6. Save everything needed for later use
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)

    print("Training done.")
    print("Saved model to:", MODEL_PATH)
    print("Saved vectorizer to:", VECTORIZER_PATH)
    print("Saved label map to:", LABEL_MAP_PATH)


if __name__ == "__main__":
    train_model()