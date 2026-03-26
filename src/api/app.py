from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

# Load model once
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}