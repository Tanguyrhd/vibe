from fastapi import FastAPI
import pickle

app = FastAPI()


@app.get('/')
def root():
    return {'Hello': 'vibe will be there soony'}

with open("../models/logistic_reg.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.get('/predict')
def predict(
    tweet='fill in your tweets'
):
    X_input=vectorizer.transform([tweet])

    pred=model.predict(X_input)[0]

    return {'MBTI personality result': str(pred)}


# then run to test LOCALLY
# uvicorn vibe_api:app --reload
