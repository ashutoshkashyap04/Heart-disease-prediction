import streamlit as st
import requests

st.title("Heart Disease Prediction")

st.write("Enter the patient's details: ")

#Input fields

age = st.number_input(
    "Age",
    min_value= 1,
    max_value= 120,
    value= 30
)

gender = st.selectbox(
    "Gender",
    ['M', 'F']
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ['ATA', 'NAP', 'ASY', 'TA']
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value= 1,
    value= 140
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value= 0,
    value= 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ['Normal', 'ST', 'LVH']
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value= 1,
    value= 150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ['N', 'Y']
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value= 0.0,
    value= 0.0
)

st_slope = st.selectbox(
    "ST Slope",
    ['Up', 'Flat', 'Down']
)

if st.button("Predict"):
    
    input_data = {
        "Age": age,
        "Sex": gender,
        "ChestPainType": chest_pain_type,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope       
    }
    
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json = input_data
    )
    
    if response.status_code == 200:
        result = response.json()
        
        if result["prediction"] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

    else:
        st.error("Prediction failed")
    
    