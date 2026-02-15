def answer_query(df, query):

    query = query.lower()

    if "columns" in query:
        return df.columns.tolist()

    if "shape" in query:
        return df.shape

    if "missing" in query:
        return df.isnull().sum().to_dict()

    if "correlation" in query:
        return df.corr().to_dict()

    return "Query not understood"
