from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.txt"
TEST_PATH = DATA_DIR / "test.txt"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"