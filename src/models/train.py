import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data.ingestion import load_data
from src.features.preprocess import preprocess_data


def train_model(data_path, target="Churn"):
    """
    Train a churn prediction model
    """

    logging.info("Loading data")
    df = load_data(data_path)

    logging.info("Preprocessing data")
    df = preprocess_data(df)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Model accuracy: {accuracy}")

    return model, accuracy