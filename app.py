import streamlit as st

st.set_page_config(
    page_title="AI Agent for Data Scientists",
    layout="wide"
)

st.title(" AI Agent for Data Scientists")

st.markdown("""
Welcome to AI Agent

This system helps you:

• Upload dataset  
• Perform EDA  
• Train ML models  
• Visualize data  
• Make predictions  
• Generate ML code  

Use the sidebar to navigate.
""")

st.success("Project initialized successfully")
