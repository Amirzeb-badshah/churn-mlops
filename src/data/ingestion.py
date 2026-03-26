import pandas as pd
import os
import logging

def load_data(path):
    """
    Load CSV data into pandas DataFrame
    """

    if not os.path.exists(path):
        raise FileNotFoundError("File not found")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        raise

    return df