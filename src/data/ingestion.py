import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("File not found")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise Exception(f"Error reading file: {e}")

    return df