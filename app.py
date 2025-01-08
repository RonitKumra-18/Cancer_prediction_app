# Import libraries
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('cancer_prediction_model.pkl')

# Streamlit app
st.title("Cancer Prediction App")
st.write("This app predicts whether a tumor is malignant or benign based on user-provided features.")

# Input fields
input_features = []
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.write("Enter the following features:")
for feature in feature_names:
    value = st.text_input(feature, "0")
    input_features.append(float(value))

# Prediction button
if st.button("Predict"):
    # Make predictions
    prediction = model.predict([input_features])
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"Prediction: {result}")
