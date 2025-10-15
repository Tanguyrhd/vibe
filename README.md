# Vibe+ – MBTI Personality Predictor

Vibe+ is a data science and NLP project that predicts MBTI personality types from text and analyzes compatibility between profiles.
However, this backend was ultimately not used, as the final model chosen was the OpenAI API, which was directly implemented in the frontend.
You can access it here: https://github.com/Tanguyrhd/vibe-frontend

## Project structure

```text
vibe/
├── data/                   # Raw, cleaned and external data
├── models/                 # Trained models (.pkl, .gguf)
├── notebooks/              # Notebooks organized by stages
│   ├── 01_data_preparation
│   ├── 02_feature_engineering
│   ├── 03_model_training
│   ├── 04_evaluation
│   └── 05_llm_mbti
├── src/
│   ├── pipeline
│   ├── prepare
│   ├── features
│   ├── models
│   ├── evaluation
│   ├── api
│   └── llm                  
├── tests/                  # Unit tests
├── Dockerfile
├── requirements.txt
├── Makefile
└── README.md
```

## Installation

Clone the repository:
```bash
git clone git@github.com:Tanguyrhd/vibe.git
cd vibe
```

Create and activate a Python environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
make reinstall_package
```

## Utilisation

Prepare the data:
```bash
make prepare
```

Build the features:
```bash
make features
```

Train the models:
```bash
make train
```

Evaluate the models:
```bash
make evaluate
```

Run the full pipeline:
```bash
make pipeline
```

## Notes
```text
- CSV files and large models are not versioned (.gitignore)
- Notebooks in archive/ are for reference only
- To add new data or models, place them in data/raw or models/
```
