def generate_cleaning_code(columns):

    return f'''
# Data Cleaning Code

import pandas as pd

df = pd.read_csv("your_dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

print(df.head())
'''


def generate_eda_code(columns):

    return f'''
# EDA Code

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_dataset.csv")

print(df.describe())

sns.heatmap(df.corr(), annot=True)
plt.show()

df.hist()
plt.show()
'''


def generate_model_code(target):

    return f'''
# Model Training Code

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop("{target}", axis=1)
y = df["{target}"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
'''


def generate_prediction_code():

    return '''
# Prediction Code

import joblib

model = joblib.load("trained_model.pkl")

prediction = model.predict(new_data)
'''


def generate_dashboard_code():

    return '''
# Dashboard Code

import streamlit as st
import seaborn as sns

st.dataframe(df)

sns.heatmap(df.corr())
st.pyplot()
'''
