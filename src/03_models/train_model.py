"""
Module to train MBTI models.
Supports multiple algorithms and saves trained models for reuse.
"""

import pickle
from pathlib import Path
from typing import Dict, Any
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

# ==========================
# SUPPRESS LIBRARY WARNINGS
# ==========================
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ==========================
# TRAINING FUNCTIONS
# ==========================

def train_logistic_regression(X_train, y_train) -> Dict[str, LogisticRegression]:
    """Train a Logistic Regression model for each binary label."""
    models = {}
    for col in y_train.columns:
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train[col])
        models[col] = clf
        print(f"✅ Logistic Regression trained for {col}")
    return models


def train_xgboost(X_train, y_train) -> Dict[str, xgb.XGBClassifier]:
    """Train an XGBoost model for each binary label."""
    models = {}
    for col in y_train.columns:
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train, y_train[col])
        models[col] = clf
        print(f"✅ XGBoost trained for {col}")
    return models


def train_lightgbm(X_train, y_train) -> Dict[str, lgb.LGBMClassifier]:
    """Train a LightGBM model for each binary label."""
    models = {}
    for col in y_train.columns:
        clf = lgb.LGBMClassifier(verbose=-1)
        clf.fit(
            X_train,
            y_train[col],
            eval_set=None,
            callbacks=[]
        )
        models[col] = clf
        print(f"✅ LightGBM trained for {col}")
    return models


# ==========================
# UTILITY FUNCTIONS
# ==========================

def evaluate_models(models: Dict[str, Any], X_test, y_test) -> None:
    """Evaluate each model on the test set and print Accuracy and F1-score."""
    for col, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test[col], y_pred)
        f1 = f1_score(y_test[col], y_pred)
        print(f"📊 {col} - Accuracy: {acc:.3f}, F1: {f1:.3f}")


def save_models(models: Dict[str, Any], model_dir: str = "models/", prefix: str = "logreg"):
    """Save each trained model as a .pkl file."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for col, model in models.items():
        file_path = Path(model_dir) / f"{prefix}_{col}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Model saved → {file_path}")


# ==========================
# MAIN PIPELINE
# ==========================

def train_all_models(X_train, X_test, y_train, y_test):
    """Train all classical models and save results."""

    print("🔹 Training Logistic Regression...")
    logreg_models = train_logistic_regression(X_train, y_train)
    evaluate_models(logreg_models, X_test, y_test)
    save_models(logreg_models, prefix="logreg")

    print("🔹 Training XGBoost...")
    xgb_models = train_xgboost(X_train, y_train)
    evaluate_models(xgb_models, X_test, y_test)
    save_models(xgb_models, prefix="xgb")

    print("🔹 Training LightGBM...")
    lgb_models = train_lightgbm(X_train, y_train)
    evaluate_models(lgb_models, X_test, y_test)
    save_models(lgb_models, prefix="lgbm")

    print("All models trained and saved.")


# ==========================
# DIRECT EXECUTION
# ==========================

if __name__ == "__main__":
    processed_folder = Path("data/processed")
    files = [f for f in processed_folder.glob("*.csv") if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No processed CSV files found in {processed_folder}")

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

    # Load pre-built train/test features
    feature_file = Path("data/processed") / f"{processed_file.stem}_tfidf_data.pkl"
    if not feature_file.exists():
        raise FileNotFoundError(f"Pre-built features not found → {feature_file}. Run build_features.py first.")

    with open(feature_file, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    train_all_models(X_train, X_test, y_train, y_test)
