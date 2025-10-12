# src/evaluation/evaluate_model.py
"""
Module pour évaluer les modèles MBTI sauvegardés.
- Charge les modèles existants
- Fait les prédictions sur un dataset
- Calcule Accuracy et F1-score
- Génère des graphiques optionnels
"""

from pathlib import Path
import pickle
from typing import Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# 🔹 CHARGEMENT DES MODELES
# ==========================

def load_models(model_dir: str = "models/", prefix: str = "logreg") -> Dict[str, Any]:
    """
    Charge tous les modèles avec un préfixe donné depuis le dossier model_dir
    """
    models = {}
    model_dir = Path(model_dir)
    for file in model_dir.glob(f"{prefix}_*.pkl"):
        label = file.stem.split("_")[-1]
        with open(file, "rb") as f:
            models[label] = pickle.load(f)
    print(f"✅ {len(models)} modèles chargés avec préfixe '{prefix}'")
    return models


# ==========================
# 🔹 PREDICTIONS
# ==========================

def predict_models(models: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """
    Fait les prédictions pour chaque modèle sur X
    """
    y_pred = pd.DataFrame(index=X.index)
    for label, model in models.items():
        y_pred[label] = model.predict(X)
    return y_pred


# ==========================
# 🔹 METRICS
# ==========================

def compute_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """
    Calcule Accuracy et F1-score pour chaque label
    """
    for col in y_true.columns:
        acc = accuracy_score(y_true[col], y_pred[col])
        f1 = f1_score(y_true[col], y_pred[col])
        print(f"📊 {col} - Accuracy: {acc:.3f}, F1: {f1:.3f}")


# ==========================
# 🔹 VISUALISATION
# ==========================

def plot_confusion(y_true: pd.DataFrame, y_pred: pd.DataFrame, label: str) -> None:
    """
    Trace la matrice de confusion pour un label donné
    """
    cm = confusion_matrix(y_true[label], y_pred[label])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# ==========================
# 🔹 PIPELINE PRINCIPAL
# ==========================

def evaluate_pipeline(X: pd.DataFrame, y: pd.DataFrame, model_dir: str = "models/", prefix: str = "logreg"):
    """
    Pipeline complet d'évaluation :
    1. Charger les modèles
    2. Faire les prédictions
    3. Calculer les métriques
    4. Afficher les matrices de confusion
    """
    models = load_models(model_dir, prefix)
    y_pred = predict_models(models, X)
    compute_metrics(y, y_pred)

    # Optionnel : afficher toutes les matrices de confusion
    for col in y.columns:
        plot_confusion(y, y_pred, col)


# ==========================
# 🔹 EXECUTION DIRECTE
# ==========================

if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv("data/processed/mbti_clean.csv")

    # Importer le vectorizer pour créer X
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    X_tfidf = vectorizer.transform(df["clean_text"])
    X = pd.DataFrame(X_tfidf.toarray(), index=df.index)
    y = df[["EI", "SN", "TF", "JP"]]

    evaluate_pipeline(X, y, prefix="logreg")
