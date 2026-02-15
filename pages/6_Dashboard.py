import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.title("📊 AI Agent Professional Dashboard")

# ===============================
# Check dataset
# ===============================

if "df" not in st.session_state:

    st.warning("Please upload dataset first")
    st.stop()

df = st.session_state["df"].copy()

# ===============================
# Dataset Overview
# ===============================

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())
col4.metric("Duplicates", df.duplicated().sum())

st.dataframe(df.head())

# ===============================
# Missing Values Chart
# ===============================

st.subheader("Missing Values Analysis")

fig, ax = plt.subplots()
df.isnull().sum().plot(kind="bar", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# ===============================
# Data Types Distribution
# ===============================

st.subheader("Data Types Distribution")

fig, ax = plt.subplots()
df.dtypes.value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
st.pyplot(fig)

# ===============================
# Numeric Feature Analysis
# ===============================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) > 0:

    st.subheader("Numeric Feature Distribution")

    selected_num = st.selectbox(
        "Select Numeric Feature",
        numeric_cols
    )

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Outlier Detection")

    fig, ax = plt.subplots()
    sns.boxplot(x=df[selected_num], ax=ax)
    st.pyplot(fig)

# ===============================
# Categorical Feature Analysis
# ===============================

cat_cols = df.select_dtypes(include="object").columns.tolist()

if len(cat_cols) > 0:

    st.subheader("Categorical Feature Distribution")

    selected_cat = st.selectbox(
        "Select Categorical Feature",
        cat_cols
    )

    fig, ax = plt.subplots()
    df[selected_cat].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ===============================
# Correlation Heatmap
# ===============================

if len(numeric_cols) > 1:

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===============================
# Scatter Plot
# ===============================

if len(numeric_cols) > 1:

    st.subheader("Feature Relationship")

    col1, col2 = st.columns(2)

    x_axis = col1.selectbox("X-axis", numeric_cols)
    y_axis = col2.selectbox("Y-axis", numeric_cols)

    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax)
    st.pyplot(fig)

# ===============================
# Model Analysis
# ===============================

model_path = "models/trained_model.pkl"
features_path = "models/feature_columns.pkl"
encoders_path = "models/label_encoders.pkl"

if not os.path.exists(model_path):

    st.warning("Train model to see analysis")
    st.stop()

if not os.path.exists(features_path):

    st.error("Feature columns file missing")
    st.stop()

if not os.path.exists(encoders_path):

    st.error("Encoders missing. Retrain model.")
    st.stop()

st.header("Model Analysis")

# Load model and encoders
model = joblib.load(model_path)
feature_columns = joblib.load(features_path)
label_encoders = joblib.load(encoders_path)

# ===============================
# Encode dataset safely
# ===============================

df_encoded = df.copy()

for col, encoder in label_encoders.items():

    if col in df_encoded.columns:

        try:

            df_encoded[col] = encoder.transform(
                df_encoded[col].astype(str)
            )

        except:

            st.warning(f"Encoding issue in column {col}")

# ===============================
# Verify columns exist
# ===============================

missing_cols = [
    col for col in feature_columns
    if col not in df_encoded.columns
]

if len(missing_cols) > 0:

    st.error(f"Missing columns: {missing_cols}")
    st.stop()

X = df_encoded[feature_columns]

# ===============================
# Detect target column
# ===============================

target_candidates = [
    col for col in df.columns
    if col not in feature_columns
]

if len(target_candidates) == 0:

    st.warning("Target column not found")
else:

    target = target_candidates[0]

    y = df[target]

    # Target Distribution
    st.subheader("Target Distribution")

    fig, ax = plt.subplots()
    sns.histplot(y, kde=True, ax=ax)
    st.pyplot(fig)

    # Predictions
    predictions = model.predict(X)

    # Actual vs Predicted
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=predictions, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    # Residual analysis
    st.subheader("Residual Analysis")

    residuals = y - predictions

    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    st.pyplot(fig)

# ===============================
# Feature Importance
# ===============================

if hasattr(model, "feature_importances_"):

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({

        "Feature": feature_columns,
        "Importance": model.feature_importances_

    }).sort_values("Importance", ascending=False)

    st.dataframe(importance_df)

    fig, ax = plt.subplots()

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax
    )

    st.pyplot(fig)

st.success("Dashboard Loaded Successfully")
from utils.code_generator import generate_dashboard_code

st.subheader("Generate Dashboard Code")

code = generate_dashboard_code()

st.code(code)

st.download_button(
    "Download Dashboard Code",
    code,
    file_name="dashboard.py"
)
