"""
Feature engineering module.
Transforms cleaned texts into numerical representations
ready for ML model training.
"""

from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

# ==========================
# FEATURE ENGINEERING FUNCTIONS
# ==========================

def create_tfidf_features(train_texts: pd.Series, test_texts: pd.Series,
                          max_features: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    """
    Transforms text data into TF-IDF vectors. Fit only on train to avoid leakage.

    Args:
        train_texts (pd.Series): Train text column
        test_texts (pd.Series): Test text column
        max_features (int): Maximum number of TF-IDF features

    Returns:
        X_train, X_test (pd.DataFrame): TF-IDF matrices
        vectorizer (TfidfVectorizer): Fitted vectorizer
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    print(f"TF-IDF features created ({X_train_tfidf.shape[1]} features)")

    return pd.DataFrame(X_train_tfidf.toarray(), index=train_texts.index), \
           pd.DataFrame(X_test_tfidf.toarray(), index=test_texts.index), \
           vectorizer


def encode_labels(df: pd.DataFrame, target_cols: list = ["EI", "SN", "TF", "JP"]) -> pd.DataFrame:
    """Extract target labels."""
    y = df[target_cols].copy()
    print(f"Labels extracted: {y.columns.tolist()}")
    return y


# ==========================
# MAIN PIPELINE
# ==========================

def build_features(df: pd.DataFrame,
                   text_col: str = "clean_text",
                   max_features: int = 5000,
                   test_size: float = 0.2,
                   vectorizer_path: str = "data/processed/vectorizer.pkl",
                   feature_path: str = "data/processed/tfidf_data.pkl") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full feature creation pipeline:
    1. Split train/test
    2. Create TF-IDF features (fit only on train)
    3. Extract labels
    4. Save the vectorizer

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    target_cols = ["EI", "SN", "TF", "JP"]
    y = encode_labels(df, target_cols)

    X_raw = df[text_col]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=42
    )

    X_train, X_test, vectorizer = create_tfidf_features(X_train_raw, X_test_raw, max_features=max_features)

    # Save the vectorizer
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved → {vectorizer_path}")

    # Save train/test features and labels
    Path(feature_path).parent.mkdir(parents=True, exist_ok=True)
    with open(feature_path, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    print(f"Train/test sets saved → {feature_path}")

    return X_train, X_test, y_train, y_test


# ==========================
# DIRECT EXECUTION
# ==========================

if __name__ == "__main__":
    processed_folder = Path("data/processed")
    files = [f for f in processed_folder.glob("*.csv") if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {processed_folder}")

    print("Available processed files:")
    for i, f in enumerate(files):
        print(f"{i + 1}: {f.name}")

    while True:
        choice = input(f"Choose the file to use (1-{len(files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            processed_file = files[int(choice) - 1]
            break
        else:
            print("❌ Invalid input, try again.")

    vectorizer_path = processed_file.parent / f"{processed_file.stem}_vectorizer.pkl"
    feature_path = Path("data/processed") / f"{processed_file.stem}_tfidf_data.pkl"

    df = pd.read_csv(processed_file)

    X_train, X_test, y_train, y_test = build_features(df, vectorizer_path=str(vectorizer_path), feature_path=str(feature_path))
