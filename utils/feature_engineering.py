import pandas as pd

def auto_feature_engineering(df):

    df_new = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        df_new[f"{col}_squared"] = df[col] ** 2

        df_new[f"{col}_log"] = df[col].apply(
            lambda x: 0 if x <= 0 else pd.np.log(x)
        )

    return df_new
