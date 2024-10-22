# Breast-Cancer-Classification
Breast Cancer Diagnosis Prediction App
Project Description
The Breast Cancer Diagnosis Prediction App is a web application built using Streamlit that allows users to input various characteristics of breast cancer tumors to predict whether the tumor is malignant (cancerous) or benign (non-cancerous). The application leverages a machine learning model trained on a dataset that includes several features related to tumor measurements.

Purpose
The primary goal of this project is to assist healthcare professionals and patients by providing an intuitive tool that can offer quick predictions regarding tumor malignancy. This can help in making informed decisions about further diagnostic procedures and treatment options.

Features
User Input: Users can enter values for various tumor characteristics through a user-friendly interface.
Real-Time Prediction: The model predicts the likelihood of malignancy based on the input features.
Data Visualization: The app can provide visual insights into the data (if implemented).
Machine Learning Model: The backend utilizes a logistic regression model (or other models, depending on your implementation) to make predictions based on trained data.
Dataset
The model is trained using a dataset that includes 25 features related to breast cancer tumors, such as:

radius_mean
texture_mean
perimeter_mean
area_mean
smoothness_mean
compactness_mean
concavity_mean
concave points_mean
symmetry_mean
radius_se
texture_se
perimeter_se
area_se
smoothness_se
compactness_se
concavity_se
concave points_se
symmetry_se
radius_worst
texture_worst
perimeter_worst
area_worst
smoothness_worst
compactness_worst
concavity_worst
concave points_worst
symmetry_worst
fractal_dimension_worst
Technical Details
The application is developed in Python and uses several libraries, including:

Streamlit: For creating the web application interface.
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
scikit-learn: For implementing the machine learning model.
Joblib: For saving and loading the trained model.
Conclusion
This Breast Cancer Diagnosis Prediction App aims to provide an accessible tool for predicting tumor malignancy, enhancing the decision-making process in healthcare. By leveraging machine learning, it seeks to improve patient outcomes through timely and informed medical advice.
