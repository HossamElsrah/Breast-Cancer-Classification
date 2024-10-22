
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and scaler
model = joblib.load('logistic_regression_model.pkl')  # replace with your model file
scaler = joblib.load('scalar.pkl')  # replace with your scaler file

# Load your dataset to get min and max values
data = pd.read_csv("data.csv")  # replace with your data file

# Title 
st.title("Breast Cancer Diagnosis Prediction")
st.subheader("This app uses a trained model to predict whether a tumor is malignant or benign based on input features.")

# User input using min and max from the dataset
radius_mean = st.slider("Radius Mean", float(data["radius_mean"].min()), float(data["radius_mean"].max()), float(data["radius_mean"].median()))  
texture_mean = st.slider("Texture Mean", float(data["texture_mean"].min()), float(data["texture_mean"].max()), float(data["texture_mean"].median()))
perimeter_mean = st.slider("Perimeter Mean", float(data["perimeter_mean"].min()), float(data["perimeter_mean"].max()), float(data["perimeter_mean"].median()))
area_mean = st.slider("Area Mean", float(data["area_mean"].min()), float(data["area_mean"].max()), float(data["area_mean"].median()))
smoothness_mean = st.slider("Smoothness Mean", float(data["smoothness_mean"].min()), float(data["smoothness_mean"].max()), float(data["smoothness_mean"].median()))
compactness_mean = st.slider("Compactness Mean", float(data["compactness_mean"].min()), float(data["compactness_mean"].max()), float(data["compactness_mean"].median()))
concavity_mean = st.slider("Concavity Mean", float(data["concavity_mean"].min()), float(data["concavity_mean"].max()), float(data["concavity_mean"].median()))
concave_points_mean = st.slider("Concave Points Mean", float(data["concave points_mean"].min()), float(data["concave points_mean"].max()), float(data["concave points_mean"].median()))
symmetry_mean = st.slider("Symmetry Mean", float(data["symmetry_mean"].min()), float(data["symmetry_mean"].max()), float(data["symmetry_mean"].median()))
radius_se = st.slider("Radius SE", float(data["radius_se"].min()), float(data["radius_se"].max()), float(data["radius_se"].median()))
perimeter_se = st.slider("Perimeter SE", float(data["perimeter_se"].min()), float(data["perimeter_se"].max()), float(data["perimeter_se"].median()))
area_se = st.slider("Area SE", float(data["area_se"].min()), float(data["area_se"].max()), float(data["area_se"].median()))
fractal_dimension_worst = st.slider("fractal dimension worst", float(data["fractal_dimension_worst"].min()), float(data["fractal_dimension_worst"].max()), float(data["fractal_dimension_worst"].median()))
compactness_se = st.slider("Compactness SE", float(data["compactness_se"].min()), float(data["compactness_se"].max()), float(data["compactness_se"].median()))
concavity_se = st.slider("Concavity SE", float(data["concavity_se"].min()), float(data["concavity_se"].max()), float(data["concavity_se"].median()))
concave_points_se = st.slider("Concave Points SE", float(data["concave points_se"].min()), float(data["concave points_se"].max()), float(data["concave points_se"].median()))
radius_worst = st.slider("Radius Worst", float(data["radius_worst"].min()), float(data["radius_worst"].max()), float(data["radius_worst"].median()))
texture_worst = st.slider("Texture Worst", float(data["texture_worst"].min()), float(data["texture_worst"].max()), float(data["texture_worst"].median()))
perimeter_worst = st.slider("Perimeter Worst", float(data["perimeter_worst"].min()), float(data["perimeter_worst"].max()), float(data["perimeter_worst"].median()))
area_worst = st.slider("Area Worst", float(data["area_worst"].min()), float(data["area_worst"].max()), float(data["area_worst"].median()))
smoothness_worst = st.slider("Smoothness Worst", float(data["smoothness_worst"].min()), float(data["smoothness_worst"].max()), float(data["smoothness_worst"].median()))
compactness_worst = st.slider("Compactness Worst", float(data["compactness_worst"].min()), float(data["compactness_worst"].max()), float(data["compactness_worst"].median()))
concavity_worst = st.slider("Concavity Worst", float(data["concavity_worst"].min()), float(data["concavity_worst"].max()), float(data["concavity_worst"].median()))
concave_points_worst = st.slider("Concave Points Worst", float(data["concave points_worst"].min()), float(data["concave points_worst"].max()), float(data["concave points_worst"].median()))
symmetry_worst = st.slider("Symmetry Worst", float(data["symmetry_worst"].min()), float(data["symmetry_worst"].max()), float(data["symmetry_worst"].median()))

# Store inputs in a DataFrame
input_data = {
    "radius_mean": radius_mean,
    "texture_mean": texture_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "smoothness_mean": smoothness_mean,
    "compactness_mean": compactness_mean,
    "concavity_mean": concavity_mean,
    "concave points_mean": concave_points_mean,
    "symmetry_mean": symmetry_mean,
    "radius_se": radius_se,
    "perimeter_se": perimeter_se,
    "area_se": area_se,
    "compactness_se": compactness_se,
    "concavity_se": concavity_se,
    "concave points_se": concave_points_se,
    "radius_worst": radius_worst,
    "texture_worst": texture_worst,
    "perimeter_worst": perimeter_worst,
    "area_worst": area_worst,
    "smoothness_worst": smoothness_worst,
    "compactness_worst": compactness_worst,
    "concavity_worst": concavity_worst,
    "concave points_worst": concave_points_worst,
    "symmetry_worst": symmetry_worst,
    "fractal_dimension_worst": fractal_dimension_worst,
}

features = pd.DataFrame(input_data, index=[0])

# Scale the input data
features_scaled = scaler.transform(features)

# Make predictions
prediction = model.predict(features_scaled)

# Display the prediction
if prediction[0] == 1:
    st.write("### Prediction: Malignant")
else:
    st.write("### Prediction: Benign")
