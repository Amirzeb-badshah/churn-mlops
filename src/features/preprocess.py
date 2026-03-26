import pandas as pd
import logging

def remove_duplicates(df):
    """Remove duplicate rows from dataframe"""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        logging.info(f"Removed {removed} duplicate rows")
    return df

def handle_missing(df):
    """Handle missing values using forward fill"""
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logging.info(f"Filling {missing_count} missing values")
        df = df.ffill()
    return df

def preprocess_data(df):
    logging.info("Starting preprocessing")

    df = remove_duplicates(df)
    df = handle_missing(df)

    logging.info("Finished preprocessing")

    return df
