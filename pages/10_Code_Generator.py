import streamlit as st
from utils.code_generator import *

st.title("💻 AI Code Generator")

if "df" not in st.session_state:

    st.warning("Upload dataset first")
    st.stop()

df = st.session_state["df"]

option = st.selectbox(

    "Select Code Type",

    [

        "Data Cleaning Code",
        "EDA Code",
        "Model Training Code",
        "Prediction Code",
        "Dashboard Code"

    ]

)

if option == "Data Cleaning Code":

    code = generate_cleaning_code(df)

elif option == "EDA Code":

    code = generate_eda_code(df)

elif option == "Model Training Code":

    target = st.selectbox("Select target column", df.columns)

    code = generate_model_code(df, target)

elif option == "Prediction Code":

    code = generate_prediction_code()

elif option == "Dashboard Code":

    code = generate_dashboard_code()

st.subheader("Generated Code")

st.code(code, language="python")

st.download_button(

    "Download Code",

    code,

    file_name="generated_code.py"

)
