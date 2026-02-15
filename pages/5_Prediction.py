import streamlit as st
import joblib
import pandas as pd
import os

st.title("🔮 Prediction")

model_path = "models/trained_model.pkl"
features_path = "models/feature_columns.pkl"
encoders_path = "models/label_encoders.pkl"

if not os.path.exists(model_path):

    st.warning("Please train model first")

else:

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    label_encoders = {}

    if os.path.exists(encoders_path):
        label_encoders = joblib.load(encoders_path)

    st.success("Model loaded successfully")

    input_data = {}

    st.subheader("Enter Feature Values")

    for feature in feature_columns:

        # If categorical feature → dropdown
        if feature in label_encoders:

            encoder = label_encoders[feature]

            options = encoder.classes_

            selected = st.selectbox(
                f"{feature}",
                options
            )

            value = encoder.transform([selected])[0]

            input_data[feature] = value

        else:

            # numeric feature
            value = st.number_input(
                f"{feature}",
                value=0.0
            )

            input_data[feature] = value

    input_df = pd.DataFrame([input_data])

    st.write("Input Preview:")
    st.dataframe(input_df)

    if st.button("Predict"):

        prediction = model.predict(input_df)

        st.success(f"Prediction: {prediction[0]}")
