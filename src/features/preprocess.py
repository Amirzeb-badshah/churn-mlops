def clean_data(df):
    # Remove missing values
    df = df.drop_duplicates()
    df = df.fillna(999)
    return df
