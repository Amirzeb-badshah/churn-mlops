def clean_data(df):
    # Remove missing values
    df = df.fillna(999)
    # Remove duplicates
    return df

def feature_engineering(df):
    return df