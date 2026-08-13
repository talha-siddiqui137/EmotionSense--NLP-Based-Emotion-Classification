import pandas as pd
from pathlib import Path


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, sep=";", header=None, names=["text", "emotion"])

    if df.isnull().values.any():
        raise ValueError(f"Dataset {path} contains missing values")

    if df.empty:
        raise ValueError(f"Dataset {path} is empty")

    return df


def build_label_map(train_df: pd.DataFrame) -> dict:
    unique_emotions = sorted(train_df["emotion"].unique())
    return {emotion: idx for idx, emotion in enumerate(unique_emotions)}


def encode_labels(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    df = df.copy()
    df["emotion"] = df["emotion"].map(label_map)

    if df["emotion"].isnull().any():
        unknown = df[df["emotion"].isnull()]
        raise ValueError(f"Found emotions not in label_map: {unknown}")

    return df