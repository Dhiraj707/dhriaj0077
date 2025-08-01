import streamlit as st
import joblib
import numpy as np
import pandas as pd


model = joblib.load("iris_model.joblib")


st.set_page_config(page_title="BMI & Iris Predictor", layout="centered")
st.title(" BMI & Iris Classification App")
st.write(" This app calculates your **BMI** and classifies **Iris flower species** using a trained machine learning model.")


st.header(" BMI Calculator")

weight = st.number_input("Enter your weight (kg)", min_value=1.0, step=0.1)
height_feet = st.number_input("Enter your height (feet)", min_value=1.0, step=0.1)

if height_feet > 0:
    height_m = height_feet / 3.28
    bmi = weight / (height_m ** 2)
    st.subheader(f"Your BMI is: **{bmi:.2f}**")

    if bmi < 16:
        st.error(" Extremely Underweight")
    elif 16 <= bmi < 18.5:
        st.warning(" Underweight")
    elif 18.5 <= bmi < 25:
        st.success("Healthy")
    elif 25 <= bmi < 30:
        st.info("Overweight")
    else:
        st.error(" Extremely Overweight")


st.header(" Iris Flower Prediction")

st.write(" Input the flower measurements to predict its species:")


sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, step=0.1, value=5.8)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, step=0.1, value=3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, step=0.1, value=4.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, step=0.1, value=1.3)


if st.button("Predict Iris Species"):
    features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(features)[0]
    species = ['Setosa', 'Versicolor', 'Virginica'][prediction]
    st.success(f" Predicted Iris species: **{species}**")
