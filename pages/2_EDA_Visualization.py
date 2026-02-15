import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis (EDA)")

# Check if dataset exists
if "df" not in st.session_state:

    st.warning("Please upload dataset first from Upload Dataset page")

else:

    df = st.session_state["df"]

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Data Types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing)

    # Missing values chart
    st.subheader("Missing Values Visualization")

    fig, ax = plt.subplots()
    missing.plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Column selection
    st.subheader("Select Column for Visualization")

    column = st.selectbox("Choose column", df.columns)

    # Histogram
    st.subheader("Histogram")

    fig, ax = plt.subplots()

    sns.histplot(df[column], kde=True, ax=ax)

    st.pyplot(fig)

    # Boxplot
    st.subheader("Boxplot")

    fig, ax = plt.subplots()

    sns.boxplot(x=df[column], ax=ax)

    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] > 1:

        fig, ax = plt.subplots(figsize=(10,6))

        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)

        st.pyplot(fig)

    else:

        st.warning("Not enough numeric columns for correlation")

    # Scatter plot
    st.subheader("Scatter Plot")

    col1 = st.selectbox("Select X-axis", numeric_df.columns)
    col2 = st.selectbox("Select Y-axis", numeric_df.columns)

    fig, ax = plt.subplots()

    sns.scatterplot(x=df[col1], y=df[col2], ax=ax)

    st.pyplot(fig)

    st.success("EDA completed successfully")
from utils.code_generator import generate_eda_code

st.subheader("Generate EDA Code")

code = generate_eda_code(df.columns)

st.code(code)

st.download_button(
    "Download EDA Code",
    code,
    file_name="eda.py"
)
