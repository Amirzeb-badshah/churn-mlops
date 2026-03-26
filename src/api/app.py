from fastapi import FastAPI
import pandas as pd

from src.models.train import train_model

app = FastAPI()

# Temporary: train model at startup
model, _ = train_model("data/churn.csv")


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    
    return {"prediction": int(prediction[0])}