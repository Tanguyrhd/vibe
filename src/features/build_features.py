# src/features/build_features.py
"""
Feature engineering module for the Vibe+ project.
Transforms cleaned texts into numerical representations
ready for ML model training.
"""

from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

# ==========================
# 🔹 FEATURE ENGINEERING FUNCTIONS
# ==========================

def create_tfidf_features(df: pd.DataFrame, text_col: str = "clean_text",
                          max_features: int = 5000) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Transforms a text column into TF-IDF vectors.

    Args:
        df (pd.DataFrame): Dataset
        text_col (str): Name of the text column
        max_features (int): Maximum number of TF-IDF features

    Returns:
        X (pd.DataFrame): TF-IDF matrix
        vectorizer (TfidfVectorizer): Fitted vectorizer object
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df[text_col])

    print(f"✅ TF-IDF features created ({X_tfidf.shape[1]} features)")

    return pd.DataFrame(X_tfidf.toarray(), index=df.index), vectorizer


def encode_labels(df: pd.DataFrame, target_cols: list = ["EI", "SN", "TF", "JP"]) -> pd.DataFrame:
    """
    Selects the binary target columns.

    Args:
        df (pd.DataFrame): Dataset
        target_cols (list): List of columns to predict

    Returns:
        y (pd.DataFrame): DataFrame of labels
    """
    y = df[target_cols].copy()
    print(f"🏷️ Labels extracted: {y.columns.tolist()}")
    return y


# ==========================
# 🔹 MAIN PIPELINE
# ==========================

def build_features(df: pd.DataFrame,
                   text_col: str = "clean_text",
                   max_features: int = 5000,
                   vectorizer_path: str = "models/vectorizer.pkl") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full feature creation pipeline:
    1. Create TF-IDF features
    2. Extract labels
    3. Save the vectorizer

    Args:
        df (pd.DataFrame): Cleaned dataset
        text_col (str): Name of the text column
        max_features (int): Maximum number of TF-IDF features
        vectorizer_path (str): Path to save the vectorizer

    Returns:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Labels
    """
    X, vectorizer = create_tfidf_features(df, text_col, max_features)
    y = encode_labels(df)

    # Save the vectorizer
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"💾 Vectorizer saved → {vectorizer_path}")

    return X, y


# ==========================
# 🔹 DIRECT EXECUTION
# ==========================

if __name__ == "__main__":
    processed_folder = Path("data/processed")
    files = [f for f in processed_folder.glob("*.csv") if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {processed_folder}")

    print("📂 Available processed files:")
    for i, f in enumerate(files):
        print(f"{i + 1}: {f.name}")

    # Ask user to choose
    while True:
        choice = input(f"Choose the file to use (1-{len(files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            processed_file = files[int(choice) - 1]
            break
        else:
            print("❌ Invalid input, try again.")

    vectorizer_path = processed_file.parent / f"{processed_file.stem}_vectorizer.pkl"
    df = pd.read_csv(processed_file)
    X, y = build_features(df, vectorizer_path=str(vectorizer_path))
