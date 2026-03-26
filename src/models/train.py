import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.data.ingestion import load_data
from src.features.preprocess import preprocess_data


def train_model(data_path):
    # Load data
    df = load_data(data_path)

    # Preprocess
    df = preprocess_data(df)

    # Example target
    target = "Churn"

    X = df.drop(columns=[target])
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model