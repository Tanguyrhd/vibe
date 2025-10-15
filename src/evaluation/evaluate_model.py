"""
Module to evaluate all saved MBTI models (Logistic Regression, LightGBM, XGBoost, etc.).

* Automatically loads all models in the models/ folder
* Makes predictions on a dataset
* Computes Accuracy and F1-score
* Optionally displays confusion matrices
"""

from pathlib import Path
import pickle
from typing import Dict, Any, List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================
# MODEL LOADING
# ==========================

def load_all_models(model_dir: str = "models/") -> Dict[str, Dict[str, Any]]:
    """
    Load all models from the given directory, grouped by model type (prefix).
    Example: {'logreg': {'EI': model, 'SN': model}, 'lgbm': {...}, 'xgb': {...}}
    """
    model_dir = Path(model_dir)
    models: Dict[str, Dict[str, Any]] = {}

    for file in model_dir.glob("*.pkl"):
        stem = file.stem  # e.g. "logreg_EI" or "lgbm_SN"
        parts = stem.split("_")
        if len(parts) < 2:
            continue

        prefix, label = parts[0], parts[-1]
        if prefix not in models:
            models[prefix] = {}

        with open(file, "rb") as f:
            models[prefix][label] = pickle.load(f)

    print(f"Loaded models by type:")
    for prefix, group in models.items():
        print(f"  - {prefix}: {len(group)} models")

    return models


# ==========================
# PREDICTIONS
# ==========================

def predict_models(models: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for each model on the provided dataset X.
    """
    y_pred = pd.DataFrame(index=X.index)
    for label, model in models.items():
        y_pred[label] = model.predict(X)
    return y_pred


# ==========================
# METRICS
# ==========================

def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, model_name: str) -> None:
    """
    Compute Accuracy and F1-score for each MBTI label.
    """
    print(f"\n===== Results for {model_name.upper()} =====")
    for col in y_true.columns:
        acc = accuracy_score(y_true[col], y_pred[col])
        f1 = f1_score(y_true[col], y_pred[col])
        print(f"{col} - Accuracy: {acc:.3f}, F1: {f1:.3f}")

# ==========================
# MAIN EVALUATION PIPELINE
# ==========================

def evaluate_all_models(X: pd.DataFrame, y: pd.DataFrame, model_dir: str = "models/"):
    """
    Evaluate all model families (logreg, lgbm, xgb, etc.)
    1. Load all models
    2. Make predictions
    3. Compute metrics
    4. Display all confusion matrices in a single figure per model family
    """
    all_models = load_all_models(model_dir)

    for model_name, models in all_models.items():
        y_pred = predict_models(models, X)
        compute_metrics(y, y_pred, model_name)

        # Create one figure with 4 subplots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()  # flatten 2D array into list

        target_cols = y.columns.tolist()

        for i, col in enumerate(target_cols):
            cm = confusion_matrix(y[col], y_pred[col])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f"{model_name.upper()} - {col}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")

    # Adjust layout for readability
    plt.tight_layout()
    plt.show()

# ==========================
# EXECUTION ENTRY POINT
# ==========================

if __name__ == "__main__":
    processed_folder = Path("data/processed")
    files = [f for f in processed_folder.glob("*.csv") if f.is_file()]

    if not files:
        raise FileNotFoundError(f"No processed CSV files found in {processed_folder}")

    print("Available processed files:")
    for i, f in enumerate(files):
        print(f"{i + 1}: {f.name}")

    # Ask user to choose the file
    while True:
        choice = input(f"Choose the file to use (1-{len(files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            processed_file = files[int(choice) - 1]
            break
        else:
            print("❌ Invalid input, try again.")

     # Load test set from pre-built features
    feature_file = Path("data/processed") / f"{processed_file.stem}_tfidf_data.pkl"
    if not feature_file.exists():
        raise FileNotFoundError(f"Pre-built features not found → {feature_file}. Run build_features.py first.")

    with open(feature_file, "rb") as f:
        _, X_test, _, y_test = pickle.load(f)

    # Evaluate all available models
    evaluate_all_models(X_test, y_test)
