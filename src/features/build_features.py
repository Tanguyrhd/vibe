# src/features/build_features.py
"""
Module de création des features pour le projet Vibe+.
Transforme les textes nettoyés en représentations numériques
prêtes pour l'entraînement des modèles ML.
"""

from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

# ==========================
# 🔹 FONCTIONS DE FEATURE ENGINEERING
# ==========================

def create_tfidf_features(df: pd.DataFrame, text_col: str = "clean_text",
                          max_features: int = 5000) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Transforme la colonne texte en vecteurs TF-IDF.

    Args:
        df (pd.DataFrame): Jeu de données
        text_col (str): Nom de la colonne texte
        max_features (int): Nombre max de features TF-IDF

    Returns:
        X (pd.DataFrame): Matrice TF-IDF
        vectorizer (TfidfVectorizer): Objet vectorizer entraîné
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df[text_col])

    print(f"✅ TF-IDF features créées ({X_tfidf.shape[1]} features)")

    return pd.DataFrame(X_tfidf.toarray(), index=df.index), vectorizer


def encode_labels(df: pd.DataFrame, target_cols: list = ["EI", "SN", "TF", "JP"]) -> pd.DataFrame:
    """
    Sélectionne les colonnes cibles binaires.

    Args:
        df (pd.DataFrame): Jeu de données
        target_cols (list): Liste des colonnes à prédire

    Returns:
        y (pd.DataFrame): DataFrame des labels
    """
    y = df[target_cols].copy()
    print(f"🏷️ Labels extraits : {y.columns.tolist()}")
    return y


# ==========================
# 🔹 PIPELINE PRINCIPAL
# ==========================

def build_features(df: pd.DataFrame,
                   text_col: str = "clean_text",
                   max_features: int = 5000,
                   vectorizer_path: str = "models/vectorizer.pkl") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline complet de création des features :
    1. Création des TF-IDF
    2. Extraction des labels
    3. Sauvegarde du vectorizer

    Args:
        df (pd.DataFrame): Données nettoyées
        text_col (str): Nom de la colonne texte
        max_features (int): Nombre max de features TF-IDF
        vectorizer_path (str): Chemin pour sauvegarder le vectorizer

    Returns:
        X (pd.DataFrame): Features
        y (pd.DataFrame): Labels
    """
    X, vectorizer = create_tfidf_features(df, text_col, max_features)
    y = encode_labels(df)

    # Sauvegarde du vectorizer
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"💾 Vectorizer sauvegardé → {vectorizer_path}")

    return X, y


# ==========================
# 🔹 EXECUTION DIRECTE
# ==========================

if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv("data/processed/mbti_clean.csv")
    X, y = build_features(df)
