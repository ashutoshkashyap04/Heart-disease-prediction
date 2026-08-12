import streamlit as st
import joblib
import pandas as pd

#Load model 
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


st.title("Heart Disease Prediction App")

st.markdown("Provide the following details")

#numerical features
age = st.number_input("Age", min_value=1, max_value=120)

resting_bp = st.number_input("Resting Blood Pressure")

cholesterol = st.number_input("Cholesterol")

max_hr = st.number_input("Maximum Heart Rate")

oldpeak = st.number_input("Old Peak", step=0.1)


# Non numerical features
fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl?",
    ["No", "Yes"]
)
fasting_bs = 1 if fasting_bs == "Yes" else 0

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ASY", "ATA", "NAP", "TA"]
)


resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "LVH", "ST"]
)


exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)


st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


if st.button("Predict"):
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == 'Yes' else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        
        "Sex_M": 1 if gender == "Male" else 0,
        
        "ChestPainType_ATA": 0,
        "ChestPainType_NAP": 0,
        "ChestPainType_TA": 0,
        
        "RestingECG_Normal": 0,
        "RestingECG_ST": 0,
        
        "ExerciseAngina_Y": 1 if exercise_angina == 'Yes' else 0,
        
        "ST_Slope_Flat": 0,
        "ST_Slope_Up": 0
    }
    
    
    #chest pain encoding
    if chest_pain == "ATA":
        input_data['ChestPainType_ATA'] = 1

    elif chest_pain == "NAP":
        input_data['ChestPainType_NAP'] = 1

    elif chest_pain == "TA":
        input_data['ChestPainType_TA'] = 1
        
        
    # Resting ECG encoding
    if resting_ecg == "Normal":
        input_data['RestingECG_Normal'] = 1

    elif resting_ecg == "ST":
        input_data['RestingECG_ST'] = 1
        
    #ST Slope Encoding
    if st_slope == "Flat":
        input_data['ST_Slope_Flat'] = 1

    elif st_slope == "Up":
        input_data['ST_Slope_Up'] = 1
        
    
    #Creating dataframe    
    input_df = pd.DataFrame([input_data])
    
    
    #Scaling numerical columns
    numeric_cols = [
        'Age',
        'RestingBP',
        'Cholesterol',
        'MaxHR',
        'Oldpeak'
    ]

    input_df[numeric_cols] = scaler.transform(
        input_df[numeric_cols]
    )
        
    #Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # display result
    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    st.write(
        f"Prediction Probability of Heart disease: {probability[0][1]:.2%}"
    )



