"""
Pipeline principal pour le projet Vibe+.
Exécute toutes les étapes de bout en bout :
1. Préparation des données
2. Feature engineering
3. Entraînement des modèles
4. Évaluation
"""

import pandas as pd
from src.data.make_dataset import clean_and_prepare_dataset
from src.features.build_features import build_features
from src.models.train_model import train_all_models
from src.evaluation.evaluate_model import evaluate_pipeline

# ==========================
# 🔹 CONFIG
# ==========================

RAW_DATA_PATH = "data/raw/mbti_clean.csv"
PROCESSED_DATA_PATH = "data/processed/mbti_clean.csv"
VECTOR_PATH = "models/vectorizer.pkl"

# ==========================
# 🔹 PIPELINE
# ==========================

def main():
    print("🔹 Étape 1 : Préparer les données")
    df = clean_and_prepare_dataset(RAW_DATA_PATH)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"💾 Données traitées sauvegardées → {PROCESSED_DATA_PATH}")

    print("🔹 Étape 2 : Créer les features")
    X, y = build_features(df, vectorizer_path=VECTOR_PATH)
    print("✅ Features créées")

    print("🔹 Étape 3 : Entraîner les modèles")
    train_all_models(X, y)
    print("✅ Modèles entraînés")

    print("🔹 Étape 4 : Évaluer les modèles")
    evaluate_pipeline(X, y, prefix="logreg")  # changer prefix pour xgb, lgbm si besoin
    print("✅ Évaluation terminée")


# ==========================
# 🔹 EXECUTION DIRECTE
# ==========================

if __name__ == "__main__":
    main()
