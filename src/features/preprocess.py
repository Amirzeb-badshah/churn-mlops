def clean_data(df):
    # Remove missing values
    df = df.fillna(method='ffill')
    # Remove duplicates
    return df