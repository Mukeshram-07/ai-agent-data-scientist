import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("🧹 Data Cleaning & Preprocessing")

# Check dataset exists
if "df" not in st.session_state:

    st.warning("Please upload dataset first")

else:

    df = st.session_state["df"]

    st.subheader("Original Dataset Shape")
    st.write(df.shape)

    st.dataframe(df.head())

    # ===============================
    # Missing Values Handling
    # ===============================

    st.subheader("Handle Missing Values")

    missing_cols = df.columns[df.isnull().any()].tolist()

    if missing_cols:

        st.write("Columns with missing values:", missing_cols)

        option = st.selectbox(
            "Select method",
            ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
        )

        if st.button("Apply Missing Value Treatment"):

            if option == "Drop rows":
                df = df.dropna()

            elif option == "Fill with mean":

                for col in df.select_dtypes(include=np.number):
                    df[col].fillna(df[col].mean(), inplace=True)

            elif option == "Fill with median":

                for col in df.select_dtypes(include=np.number):
                    df[col].fillna(df[col].median(), inplace=True)

            elif option == "Fill with mode":

                for col in df.columns:
                    df[col].fillna(df[col].mode()[0], inplace=True)

            st.session_state["df"] = df

            st.success("Missing values handled")

    else:

        st.success("No missing values found")

    # ===============================
    # Remove duplicates
    # ===============================

    st.subheader("Duplicate Removal")

    duplicates = df.duplicated().sum()

    st.write("Duplicate rows:", duplicates)

    if st.button("Remove Duplicates"):

        df = df.drop_duplicates()

        st.session_state["df"] = df

        st.success("Duplicates removed")

    # ===============================
    # Encode categorical data
    # ===============================

    st.subheader("Encode Categorical Columns")

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    st.write("Categorical columns:", cat_cols)

    if st.button("Encode Categorical Columns"):

        le = LabelEncoder()

        for col in cat_cols:

            df[col] = le.fit_transform(df[col].astype(str))

        st.session_state["df"] = df

        st.success("Encoding completed")

    # ===============================
    # Show cleaned dataset
    # ===============================

    st.subheader("Cleaned Dataset")

    st.write(df.shape)

    st.dataframe(df.head())

    # ===============================
    # Download cleaned dataset
    # ===============================

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned Dataset",
        csv,
        "cleaned_dataset.csv",
        "text/csv"
    )
from utils.code_generator import generate_cleaning_code

st.subheader("Generate Cleaning Code")

code = generate_cleaning_code(df.columns)

st.code(code, language="python")

st.download_button(
    "Download Cleaning Code",
    code,
    file_name="cleaning.py"
)
