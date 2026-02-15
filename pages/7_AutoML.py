import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

st.title("🤖 AI Agent AutoML Engine")

# ===============================
# Check dataset
# ===============================

if "df" not in st.session_state:

    st.warning("Please upload dataset first")
    st.stop()

df = st.session_state["df"].copy()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ===============================
# Select target column
# ===============================

target = st.selectbox("Select Target Column", df.columns)

# ===============================
# Run AutoML
# ===============================

if st.button("Run AutoML"):

    df_encoded = df.copy()

    label_encoders = {}

    # Encode categorical columns
    for col in df_encoded.columns:

        if df_encoded[col].dtype == "object":

            le = LabelEncoder()

            df_encoded[col] = le.fit_transform(
                df_encoded[col].astype(str)
            )

            label_encoders[col] = le

    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Detect problem type
    classification = y.nunique() <= 10

    results = {}

    st.subheader("Testing Multiple Models...")

    if classification:

        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier()

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds)

            results[name] = score

    else:

        models = {

            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor()

        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            score = r2_score(y_test, preds)

            results[name] = score

    # ===============================
    # Model comparison
    # ===============================

    results_df = pd.DataFrame({

        "Model": list(results.keys()),
        "Score": list(results.values())

    }).sort_values("Score", ascending=False)

    st.subheader("Model Comparison Results")

    st.dataframe(results_df)

    best_model_name = results_df.iloc[0]["Model"]

    best_model = models[best_model_name]

    st.success(f"Best Model Selected: {best_model_name}")

    # ===============================
    # Save best model
    # ===============================

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")

    st.success("Best model saved successfully")

    st.session_state["best_model"] = best_model

    # ===============================
    # Confusion Matrix
    # ===============================

    if classification:

        st.subheader("Confusion Matrix")

        preds = best_model.predict(X_test)

        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

    # ===============================
    # Model Download
    # ===============================

    st.subheader("Download Best Model")

    with open("models/best_model.pkl", "rb") as f:

        st.download_button(
            "Download Model",
            f,
            file_name="best_model.pkl"
        )

    # ===============================
    # Code Generator
    # ===============================

    st.subheader("Generated Training Code")

    code = f"""
# AI Agent Generated Code

from sklearn.model_selection import train_test_split
from sklearn.ensemble import {best_model.__class__.__name__}

model = {best_model.__class__.__name__}()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
"""

    st.code(code, language="python")

    # ===============================
    # Dataset Insights Generator
    # ===============================

    st.subheader("Automatic Dataset Insights")

    corr = df_encoded.corr()

    for col in corr.columns:

        strongest = corr[col].drop(col).abs().idxmax()

        value = corr[col][strongest]

        st.write(
            f"{col} strongly correlates with {strongest} "
            f"(correlation: {value:.2f})"
        )

st.success("AutoML Engine Ready")
