import streamlit as st
import pickle
import numpy as np

# =========================
# Load Pickled Model
# =========================
with open("rf_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# UI Layout
# =========================
st.set_page_config(page_title="Cancer Risk Prediction", layout="centered")

st.title("🧬 Cancer Risk Prediction System")
st.write("Enter patient details to predict cancer risk")

# =========================
# User Inputs
# =========================
age = st.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=18.5)

gender = st.selectbox("Gender", options=["Male", "Female"])
smoking = st.selectbox("Smoking Status", options=["No", "Yes"])
history = st.selectbox("Cancer History", options=["No", "Yes"])

genetic_risk = st.selectbox("Genetic Risk Level", options=[0, 1, 2])
alcohol_intake = st.number_input("Alcohol Intake (per week)", min_value=0.0, value=0.0)
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, value=3.0)

# =========================
# Convert Inputs
# =========================
gender_val = 0 if gender == "Male" else 1
smoking_val = 1 if smoking == "Yes" else 0
history_val = 1 if history == "Yes" else 0

# =========================
# Prediction Button
# =========================
if st.button("🔍 Predict"):
    user_data = np.array([[
        age,
        bmi,
        gender_val,
        smoking_val,
        history_val,
        genetic_risk,
        alcohol_intake,
        physical_activity
    ]])

    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)[0]
    probability = model.predict_proba(user_data_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error(f"⚠️ **Cancer Risk Detected**\n\nProbability: **{probability:.2f}**")
    else:
        st.success(f"✅ **Low Cancer Risk**\n\nProbability: **{1 - probability:.2f}**")
