import pandas as pd
import logging
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data.ingestion import load_data
from src.features.preprocess import preprocess_data


logging.basicConfig(level=logging.INFO)


def train_model(data_path, target="Churn"):
    df = load_data(data_path)
    df = preprocess_data(df)

    # Drop non-numeric columns (except target)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    X = df[numeric_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Accuracy: {accuracy}")

    return model


def save_model(model):
    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)


# 👇 THIS RUNS WHEN YOU EXECUTE FILE
if __name__ == "__main__":
    model = train_model("data/sample.csv")
    save_model(model)