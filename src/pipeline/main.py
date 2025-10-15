"""
Main pipeline for the Vibe+ project.
Runs all steps end-to-end:
1. Data preparation
2. Feature engineering
3. Model training
4. Model evaluation
"""

import os
import pandas as pd
from pathlib import Path
from src.prepare.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_all_models
from src.evaluation.evaluate_model import evaluate_all_models

# ==========================
# 🔹 CONFIGURATION
# ==========================

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_PATH = "data/processed/processed_dataset.csv"
FEATURES_DIR = "data/processed/features"
MODELS_DIR = "models/"
RESULTS_DIR = "results/"

# ==========================
# 🔹 PIPELINE
# ==========================

def main():
    print("🔹 Step 1: Choose the raw data...")
    raw_folder = Path("data/raw")
    files = [f for f in raw_folder.glob("*.csv") if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_folder}")
    print("Available files in data/raw/:")
    for i, f in enumerate(files):
        print(f"{i + 1}: {f.name}")

    while True:
        choice = input(f"Select the file to use (1-{len(files)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            raw_file = files[int(choice) - 1]
            break
        else:
            print("❌ Invalid input, try again.")

    print("🔹 Step 2: Preparing data...")
    processed_file = Path("data/processed") / f"{raw_file.stem}_clean.csv"
    df = make_dataset(raw_path=str(raw_file), processed_path=str(processed_file))
    print(f" Clean data saved")

    print("🔹 Step 3: Building features...")
    X_train, X_test, y_train, y_test = build_features(df)
    print(" Features successfully built and saved")

    print("🔹 Step 4: Training models...")
    train_all_models(X_train, X_test, y_train, y_test)
    print(" All models trained and saved")

    print("🔹 Step 5: Evaluating models...")
    evaluate_all_models(X_test, y_test)
    print(" Evaluation completed successfully")


# ==========================
# 🔹 DIRECT EXECUTION
# ==========================

if __name__ == "__main__":
    main()
