from fastapi import FastAPI
import joblib
from pathlib import Path
import pandas as pd

from backend.schemas import HeartDiseaseInput

app = FastAPI(title = "Heart Disease Prediction API")

# load saved objects
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "stacking_classifier_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
columns = joblib.load(BASE_DIR / "models" / "columns.pkl")

numeric_cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak"
]

@app.get('/')
def home():
    return {
        "Message" : "Heart Disease Prediction"
    }

@app.post('/predict')
def predict(data : HeartDiseaseInput):
    
    #convert pydantic input to dataframe
    input_df = pd.DataFrame([data.model_dump()])
    
    # One Hot Encoding
    input_df = pd.get_dummies(input_df)
    
    # Making sure that input has exactly the same column used during training
    input_df = input_df.reindex(
        columns= columns,
        fill_value= 0
    )
    
    # Standardization
    input_df[numeric_cols] = scaler.transform(
        input_df[numeric_cols]
    )
    
    #prediction
    prediction = model.predict(input_df)[0]
    
    return {
        "prediction" : int(prediction)
    }
