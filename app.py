import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Lung Capacity Prediction", page_icon="🫁")

st.title("🫁 Lung Capacity Prediction")
st.write("Enter the patient's details below.")

# Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=16)

height = st.number_input(
    "Height (inches)",
    min_value=20.0,
    max_value=90.0,
    value=57.0
)

gender = st.selectbox("Gender", ["Male", "Female"])
smoke = st.selectbox("Smoke", ["No", "Yes"])

# Convert categorical values
gender = 1 if gender == "Male" else 0
smoke = 1 if smoke == "Yes" else 0

if st.button("Predict Lung Capacity"):

    features = np.array([[age, height, gender, smoke]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    st.success(f"Predicted Lung Capacity: **{prediction[0]:.2f}**")
