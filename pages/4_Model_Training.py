import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ===============================
# Ensure models folder exists
# ===============================

os.makedirs("models", exist_ok=True)

st.title("🤖 Model Training")

# ===============================
# Check dataset exists
# ===============================

if "df" not in st.session_state:

    st.warning("Please upload dataset first")

else:

    # Copy dataset safely
    df = st.session_state["df"].copy()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    # ===============================
    # Select target column
    # ===============================

    target = st.selectbox("Select Target Column", df.columns)

    # ===============================
    # Train button
    # ===============================

    if st.button("Train Model"):

        try:

            # ===============================
            # Remove ID-like columns safely
            # ===============================

            id_cols = [
                col for col in df.columns
                if ("id" in col.lower()) and col != target
            ]

            if id_cols:

                df.drop(columns=id_cols, inplace=True)

                st.info(f"Removed ID columns: {id_cols}")

            # ===============================
            # Encode categorical columns
            # ===============================

            label_encoders = {}

            for col in df.columns:

                if df[col].dtype == "object":

                    le = LabelEncoder()

                    df[col] = le.fit_transform(df[col].astype(str))

                    label_encoders[col] = le

            # Save encoders (CRITICAL FIX)
            joblib.dump(label_encoders, "models/label_encoders.pkl")

            st.success("Categorical encoding completed and encoders saved")

            # ===============================
            # Split features and target
            # ===============================

            X = df.drop(target, axis=1)
            y = df[target]

            st.write("Feature columns:", list(X.columns))

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            st.success("Data split completed")

            # ===============================
            # Detect problem type
            # ===============================

            if y.nunique() <= 10:

                st.info("Detected Classification Problem")

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)

                st.success(f"Accuracy: {accuracy:.4f}")

            else:

                st.info("Detected Regression Problem")

                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

                model.fit(X_train, y_train)

                predictions = model.predict(X_test)

                r2 = r2_score(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)

                st.success(f"R2 Score: {r2:.4f}")
                st.write(f"MSE: {mse:.4f}")

            # ===============================
            # Save model and feature columns
            # ===============================

            joblib.dump(model, "models/trained_model.pkl")

            joblib.dump(
                X.columns.tolist(),
                "models/feature_columns.pkl"
            )

            st.session_state["model"] = model
            st.session_state["feature_columns"] = X.columns.tolist()

            st.success("Model trained and saved successfully")

            # ===============================
            # Show feature importance
            # ===============================

            if hasattr(model, "feature_importances_"):

                importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)

                st.subheader("Feature Importance")

                st.dataframe(importance_df)

        except Exception as e:

            st.error(f"Training Error: {e}")
from utils.code_generator import generate_model_code

st.subheader("Generate Model Code")

code = generate_model_code(target)

st.code(code)

st.download_button(
    "Download Model Code",
    code,
    file_name="model.py"
)
