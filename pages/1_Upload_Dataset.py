import streamlit as st
import pandas as pd

st.title("📂 Upload Dataset")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    
    df = pd.read_csv(file)
    
    st.session_state["df"] = df
    
    st.success("Dataset uploaded successfully")
    
    st.dataframe(df.head())
