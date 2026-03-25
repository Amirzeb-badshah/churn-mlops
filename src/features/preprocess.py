def clean_data(df):
    # Remove missing values
    df = df.fillna("Unknown")
    # Remove duplicates
    return df

def feature_engineering(df):
    return df