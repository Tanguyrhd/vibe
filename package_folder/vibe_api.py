from fastapi import FastAPI
import pickle

app = FastAPI()


@app.get('/')
def root():
    return {'Hello': 'vibe will be there soony'}


@app.get('/predict')
def predict(
    input1=1,
    input2=1,
    input3=1,
    input4=1,
):

    with open('../models/best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    pred = model.predict([[input1,input2,input3,input4]])[0]

    return {'prediction': float(pred)}


# then run to test LOCALLY
# uvicorn vibe_api:app --reload
