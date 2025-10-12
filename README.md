# Vibe+ – MBTI Personality Predictor

**Vibe+** est un projet de data science et NLP qui prédit les types de personnalité MBTI à partir de textes et analyse la compatibilité entre profils.

## Structure du projet

```text
vibe/
├── data/                   # Données brutes, nettoyées et externes
├── models/                 # Modèles entraînés (.pkl, .gguf)
├── notebooks/              # Notebooks organisés par étapes
│   ├── 01_data_preparation
│   ├── 02_feature_engineering
│   ├── 03_model_training
│   ├── 04_evaluation
│   ├── 05_llm_mbti
│   └── archive             # Anciennes expérimentations
├── src/                    # Modules Python
│   ├── api/                # API
│   ├── data/               # Préparation datasets
│   ├── features/           # Construction features
│   ├── llm/                # LLM (OpenAI, LLaMA)
│   ├── models/             # Entraînement et save/load
│   └── pipeline/           # Exécution projet complet
├── tests/                  # Tests unitaires
├── Dockerfile
├── requirements.txt
├── Makefile
└── README.md
```

## Installation

Cloner le dépôt :  
```bash
git clone git@github.com:Tanguyrhd/vibe.git
cd vibe
```

Créer un environnement Python et activer :
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

Installer les dépendances :
```bash
pip install -r requirements.txt
```

Construire le Docker container (optionnel) :
```bash
docker build -t vibe .
```

## Usage
Préparer les données :
```bash
python -m src.data.make_dataset
```

Construire les features :
```bash
python -m src.features.build_features
```

Entraîner les modèles :
```bash
python -m src.models.train_model
```

Évaluer les modèles :
```bash
python -m src.evaluation.evaluate_model
```

Exécuter le pipeline complet :
```bash
python -m src.pipeline.main
```

## Notes
```text
- Les fichiers CSV et modèles lourds ne sont pas versionnés (.gitignore)
- Les notebooks dans archive/ sont pour référence seulement
- Pour ajouter de nouvelles données ou modèles, place-les dans data/raw ou models/
```
