def clean_data(df):
    # Remove missing values
    df = df.fillna(0)
    # Remove duplicates
    return df