import pandas as pd
import logging

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_missing(df):
    return df.fillna(method="ffill")

def preprocess_data(df):
    logging.info("Starting preprocessing")

    df = remove_duplicates(df)
    df = handle_missing(df)

    logging.info("Finished preprocessing")

    return df