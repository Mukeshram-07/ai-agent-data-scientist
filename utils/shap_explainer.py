import shap
import pandas as pd
import numpy as np

def generate_shap_values(model, X):

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Select correct explainer based on model type

    model_name = model.__class__.__name__

    # Tree models
    if model_name in [
        "RandomForestClassifier",
        "RandomForestRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor"
    ]:

        explainer = shap.TreeExplainer(model)

    # Linear models (LogisticRegression, LinearRegression)
    elif model_name in [
        "LogisticRegression",
        "LinearRegression"
    ]:

        explainer = shap.LinearExplainer(model, X)

    # Fallback for other models
    else:

        explainer = shap.KernelExplainer(
            model.predict,
            shap.sample(X, 100)
        )

    shap_values = explainer(X)

    return shap_values, explainer
