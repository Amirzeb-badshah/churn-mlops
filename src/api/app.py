from fastapi import FastAPI
import pandas as pd
import pickle
import logging

app = FastAPI()

# Load model once
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}