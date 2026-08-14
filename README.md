# ❤️ Heart Disease Prediction

An end-to-end **Machine Learning application for heart disease prediction**, built using Scikit-learn and deployed as a production-style web application.

The project covers the complete machine learning workflow:

**Data Loading → EDA → Train/Test Split → Preprocessing → Model Training → Model Evaluation → Hyperparameter Tuning → Ensemble Learning → Model Serialization → FastAPI API → Cloud Deployment → Streamlit Frontend**

The final system allows users to enter patient-related clinical information through a web interface and receive a heart disease prediction from a trained **Stacking Classifier**.

---

## 🚀 Live Application

### 🌐 Web Application

The frontend is built using **Streamlit** and deployed on **Streamlit Cloud**.

> Add your Streamlit Cloud URL here:
>
> `https://your-streamlit-app-url.streamlit.app/`

### ⚡ Backend API

The prediction API is built using **FastAPI** and deployed on **Render**.

> Backend API:
>
> `https://heart-disease-prediction-api.onrender.com`

The frontend communicates with the FastAPI backend through the `/predict` endpoint.

---

# 📌 Project Overview

Heart disease is one of the major health concerns worldwide. Machine learning can be used to analyze clinical attributes and identify patterns associated with the presence or absence of heart disease.

This project builds a binary classification system that predicts whether a patient is likely to have heart disease based on clinical and demographic features.

The project does not stop at training a single machine learning model. Multiple classification algorithms were trained and evaluated, the strongest models were selected using **F1 Score**, hyperparameter tuning was performed, and the best-performing models were combined using a **Stacking Classifier**.

The final trained model is exposed through a REST API using FastAPI, while a Streamlit application provides the user interface.

---

# 🎯 Objectives

The main objectives of this project are:

* Perform exploratory data analysis on the heart disease dataset.
* Clean and prepare the raw dataset.
* Create separate training and testing datasets.
* Apply categorical feature encoding.
* Standardize numerical features.
* Save preprocessing artifacts for inference.
* Train and compare multiple classification algorithms.
* Select models primarily based on F1 Score.
* Perform hyperparameter tuning on the best-performing models.
* Build a Stacking Classifier using the strongest tuned models.
* Evaluate the final ensemble model.
* Serialize the trained model and preprocessing objects.
* Build a REST API using FastAPI.
* Deploy the backend API on Render.
* Build an interactive frontend using Streamlit.
* Deploy the frontend using Streamlit Cloud.

---

# 🏗️ Project Architecture

The overall architecture of the project is:

```text
                         ┌──────────────────────┐
                         │      User            │
                         │  Clinical Inputs     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Streamlit Frontend  │
                         │      Streamlit       │
                         └──────────┬───────────┘
                                    │
                              HTTP POST
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    FastAPI Backend   │
                         │      /predict        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Preprocessing     │
                         │ Encoding + Scaling   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Stacking Classifier  │
                         │   Trained ML Model   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Prediction       │
                         │ Heart Disease:      │
                         │      Yes / No        │
                         └──────────────────────┘
```

---

# 📂 Project Structure

```text
Heart-disease-prediction/
│
├── backend/
│   ├── app.py
│   └── schemas.py
│
├── data/
│   ├── processed/
│   │   ├── test.csv
│   │   └── train.csv
│   │
│   └── raw/
│       ├── heart.csv
│       ├── test.csv
│       └── train.csv
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── columns.pkl
│   ├── scaler.pkl
│   └── stacking_classifier_model.pkl
│
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training_and_evaluation.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 📊 Dataset

The project uses a heart disease dataset containing clinical and demographic attributes of patients.

The target variable is:

```text
HeartDisease
```

The target represents whether heart disease is present.

The dataset contains the following input features:

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `Age`            | Age of the patient                    |
| `Sex`            | Biological sex                        |
| `ChestPainType`  | Type of chest pain                    |
| `RestingBP`      | Resting blood pressure                |
| `Cholesterol`    | Serum cholesterol                     |
| `FastingBS`      | Fasting blood sugar indicator         |
| `RestingECG`     | Resting electrocardiographic result   |
| `MaxHR`          | Maximum heart rate achieved           |
| `ExerciseAngina` | Exercise-induced angina               |
| `Oldpeak`        | ST depression induced by exercise     |
| `ST_Slope`       | Slope of the peak exercise ST segment |

Target:

| Target | Meaning          |
| ------ | ---------------- |
| `0`    | No heart disease |
| `1`    | Heart disease    |

---

# 🔄 Machine Learning Workflow

The complete ML pipeline is divided into three notebooks.

```text
01_data_loading_and_eda.ipynb
              ↓
        Data Cleaning
              ↓
        Exploratory Analysis
              ↓
        Train/Test Split
              ↓
02_data_preprocessing.ipynb
              ↓
      One-Hot Encoding
              ↓
       Standardization
              ↓
       Save Artifacts
              ↓
03_model_training_and_evaluation.ipynb
              ↓
       Model Comparison
              ↓
      Model Selection
              ↓
   Hyperparameter Tuning
              ↓
     Stacking Classifier
              ↓
      Final Evaluation
              ↓
      Model Serialization
```

---

# 1️⃣ Data Loading and EDA

Notebook:

```text
notebooks/01_data_loading_and_eda.ipynb
```

The original dataset was loaded from:

```text
data/raw/heart.csv
```

Basic data cleaning and exploratory data analysis were performed.

The dataset was then divided into training and testing datasets.

The resulting files were saved as:

```text
data/raw/train.csv
data/raw/test.csv
```

This separation allows the preprocessing and modeling stages to operate on independent training and testing datasets.

---

# 2️⃣ Data Preprocessing

Notebook:

```text
notebooks/02_data_preprocessing.ipynb
```

The preprocessing pipeline consists primarily of:

1. Categorical feature encoding
2. Numerical feature standardization
3. Saving preprocessing artifacts

---

## 🔹 One-Hot Encoding

Categorical features were converted into numerical representation using Pandas `get_dummies()`.

```python
train_df = pd.get_dummies(
    train_df,
    drop_first=True
)

train_df = train_df.astype("int")

test_df = pd.get_dummies(
    test_df,
    drop_first=True
)

test_df = test_df.astype("int")
```

`drop_first=True` was used to avoid redundant dummy variables.

The categorical variables are therefore transformed into numerical columns that can be consumed by machine learning algorithms.

---

## 🔹 Feature Standardization

The following numerical features were standardized:

```text
Age
RestingBP
Cholesterol
MaxHR
Oldpeak
```

`StandardScaler` from Scikit-learn was used.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numeric_cols = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak"
]

train_df[numeric_cols] = scaler.fit_transform(
    train_df[numeric_cols]
)

test_df[numeric_cols] = scaler.transform(
    test_df[numeric_cols]
)
```

### Important preprocessing principle

The scaler is **fitted only on the training data**:

```python
scaler.fit_transform(train_df[numeric_cols])
```

The same fitted scaler is then applied to the test data:

```python
scaler.transform(test_df[numeric_cols])
```

This prevents information from the test dataset from influencing the preprocessing process.

---

## 🔹 Processed Dataset

After preprocessing, the datasets were saved as:

```text
data/processed/train.csv
data/processed/test.csv
```

---

# 💾 Preprocessing Artifacts

Two preprocessing artifacts were saved in the `models/` directory:

```text
models/
├── columns.pkl
└── scaler.pkl
```

### `scaler.pkl`

Contains the fitted `StandardScaler`.

During inference, the same scaler is used to transform numerical user inputs.

### `columns.pkl`

Contains the feature-column structure required by the trained model.

This is particularly important because the user-facing application does not directly provide one-hot encoded columns.

For example, the user enters:

```text
Sex = M
ChestPainType = ATA
RestingECG = Normal
```

The backend converts these values into the exact feature representation expected by the trained model.

This keeps the frontend simple while ensuring that the model receives the correct input format.

---

# 3️⃣ Model Training and Evaluation

Notebook:

```text
notebooks/03_model_training_and_evaluation.ipynb
```

The processed training and testing datasets were loaded and separated into:

```text
X_train
X_test
y_train
y_test
```

Nine classification algorithms were initially trained and evaluated.

---

# 🤖 Models Evaluated

The following models were compared:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree
5. Random Forest
6. Support Vector Machine
7. AdaBoost
8. Gradient Boosting
9. XGBoost

For SVM, probability estimation was explicitly enabled:

```python
SVC(probability=True)
```

This was required because probability estimates were used for metrics such as ROC-AUC and later ensemble modeling.

---

# 📈 Initial Model Comparison

The primary selection metric was **F1 Score**.

This was chosen because it provides a balance between precision and recall and is useful when both false positives and false negatives matter.

### Initial Results

| Rank | Model               | Accuracy | Precision | Recall |   F1 Score | ROC-AUC |
| ---: | ------------------- | -------: | --------: | -----: | ---------: | ------: |
|    1 | AdaBoost            |   0.8967 |    0.9109 | 0.9020 | **0.9064** |  0.9241 |
|    2 | Random Forest       |   0.8804 |    0.8922 | 0.8922 | **0.8922** |  0.9305 |
|    3 | Logistic Regression |   0.8750 |    0.8835 | 0.8922 | **0.8878** |  0.9333 |
|    4 | Gradient Boost      |   0.8696 |    0.8824 | 0.8824 | **0.8824** |  0.9100 |
|    5 | Naive Bayes         |   0.8696 |    0.9063 | 0.8529 | **0.8788** |  0.9443 |
|    6 | SVM                 |   0.8641 |    0.8812 | 0.8725 | **0.8768** |  0.9346 |
|    7 | KNN                 |   0.8533 |    0.8641 | 0.8725 | **0.8683** |  0.9286 |
|    8 | XGBoost             |   0.8315 |    0.8586 | 0.8333 | **0.8458** |  0.9116 |
|    9 | Decision Tree       |   0.7337 |    0.8118 | 0.6765 | **0.7380** |  0.7407 |

---

# 🏆 Selecting the Top 6 Models

Based on F1 Score, the top six models were selected for further optimization:

```text
1. AdaBoost
2. Random Forest
3. Logistic Regression
4. Gradient Boosting
5. Naive Bayes
6. SVM
```

The remaining three models were not taken forward to the hyperparameter tuning stage.

---

# ⚙️ Hyperparameter Tuning

The selected six models were individually tuned using hyperparameter optimization.

After tuning, their F1 Scores were compared again.

### Tuned Model Results

| Rank | Model               |     F1 Score |
| ---: | ------------------- | -----------: |
|    1 | **Random Forest**   | **0.886029** |
|    2 | SVM                 |     0.875190 |
|    3 | AdaBoost            |     0.873734 |
|    4 | Gradient Boost      |     0.870785 |
|    5 | Logistic Regression |     0.868857 |
|    6 | Naive Bayes         |     0.854488 |

The tuned **Random Forest** achieved the highest F1 Score among the six tuned models.

---

# 🧩 Stacking Classifier

Instead of using the best individual model directly, the three strongest tuned models were selected and combined using a **Stacking Classifier**.

The selected base models were:

```text
Random Forest
SVM
AdaBoost
```

A Logistic Regression model was used as the final estimator.

Conceptually:

```text
                 ┌─────────────────┐
                 │  Random Forest  │
                 └────────┬────────┘
                          │
                          │
Input ────────────────────┼──────────────┐
                          │              │
                 ┌────────▼────────┐     │
                 │      SVM        │     │
                 └────────┬────────┘     │
                          │              │
                          │              │
                 ┌────────▼────────┐     │
                 │    AdaBoost     │     │
                 └────────┬────────┘     │
                          │              │
                          ▼              │
                 ┌─────────────────┐     │
                 │ Logistic        │     │
                 │ Regression      │◄────┘
                 │ Final Estimator │
                 └────────┬────────┘
                          │
                          ▼
                     Prediction
```

The purpose of stacking is to combine the strengths of multiple different models rather than relying on a single classifier.

---

# 📊 Final Model Performance

The final Stacking Classifier achieved:

### Classification Report

|                Class | Precision | Recall | F1 Score | Support |
| -------------------: | --------: | -----: | -------: | ------: |
|                    0 |      0.87 |   0.87 |     0.87 |      82 |
|                    1 |      0.89 |   0.89 |     0.89 |     102 |
| **Overall Accuracy** |           |        | **0.88** | **184** |

### Additional Metrics

```text
Accuracy:  0.88
ROC-AUC:   0.93675
```

The ROC-AUC score of approximately **0.937** indicates strong discriminatory performance on the evaluation dataset.

---

# 💾 Model Serialization

After final evaluation, the trained Stacking Classifier was serialized and saved as:

```text
models/stacking_classifier_model.pkl
```

The `models/` directory therefore contains:

```text
models/
├── columns.pkl
├── scaler.pkl
└── stacking_classifier_model.pkl
```

These artifacts allow the trained machine learning pipeline to be reused during API inference without retraining the models.

---

# ⚡ FastAPI Backend

After completing the machine learning pipeline, a REST API was developed using **FastAPI**.

Backend structure:

```text
backend/
├── app.py
└── schemas.py
```

### `backend/app.py`

Responsible for:

* Loading the trained model
* Loading the scaler
* Loading the feature columns
* Receiving prediction requests
* Converting user-friendly inputs into model-compatible features
* Applying preprocessing
* Generating predictions
* Returning the prediction response

### `backend/schemas.py`

Contains Pydantic schemas used for validating incoming API data.

---

# 🔄 Prediction API Workflow

The backend prediction flow is:

```text
User Input
    ↓
Pydantic Validation
    ↓
Categorical Encoding
    ↓
Feature Alignment
    ↓
Numerical Scaling
    ↓
Stacking Classifier
    ↓
Prediction
    ↓
API Response
```

The API accepts the original human-readable clinical features rather than requiring users to manually provide one-hot encoded columns.

---

# 🧑‍⚕️ User Input Features

The frontend collects the following information:

```python
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
```

The user therefore interacts with meaningful clinical fields rather than the transformed machine-learning feature columns.

---

# 🌐 Frontend

The frontend was developed using **Streamlit**.

Structure:

```text
frontend/
└── app.py
```

The application provides an interactive form where the user enters the required patient information.

The frontend sends the entered information to the deployed FastAPI backend through an HTTP POST request.

The API processes the request and returns the prediction.

---

# ☁️ Deployment Architecture

The application is split into two independently deployed services.

```text
                    INTERNET
                       │
                       ▼
        ┌──────────────────────────┐
        │    Streamlit Cloud       │
        │                          │
        │     Frontend             │
        └────────────┬─────────────┘
                     │
                     │ HTTP Request
                     ▼
        ┌──────────────────────────┐
        │         Render           │
        │                          │
        │      FastAPI Backend     │
        │                          │
        │       /predict           │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │       ML Artifacts       │
        │                          │
        │  Stacking Classifier     │
        │  StandardScaler          │
        │  Feature Columns         │
        └──────────────────────────┘
```

---

# 🚀 Backend Deployment — Render

The FastAPI backend was deployed on **Render**.

The backend exposes the prediction endpoint:

```text
POST /predict
```

The deployed API is accessed by the Streamlit frontend.

The frontend does not perform the machine learning prediction itself. Instead, it sends the input data to the remote FastAPI server.

---

# 🎨 Frontend Deployment — Streamlit Cloud

The Streamlit application was deployed separately using Streamlit Cloud.

This architecture provides a clean separation between:

```text
Frontend
    ↓
API
    ↓
Machine Learning Model
```

This separation also makes it possible to update the frontend and backend independently.

---

# 🔌 API Request Flow

A typical prediction request follows this process:

### Step 1 — User enters data

The user fills the Streamlit form.

Example:

```text
Age             → 55
Sex             → M
ChestPainType   → ATA
RestingBP       → 140
Cholesterol     → 250
FastingBS       → 0
RestingECG      → Normal
MaxHR           → 150
ExerciseAngina  → N
Oldpeak         → 1.2
ST_Slope        → Up
```

### Step 2 — Frontend creates JSON

The Streamlit application converts the form values into a JSON request.

### Step 3 — Request is sent to FastAPI

The frontend sends the request to:

```text
POST /predict
```

### Step 4 — Backend preprocesses input

The backend:

* Encodes categorical variables.
* Aligns the resulting features with `columns.pkl`.
* Applies the saved `StandardScaler`.
* Passes the final feature vector to the trained Stacking Classifier.

### Step 5 — Model generates prediction

The serialized model generates the prediction.

### Step 6 — Result is returned

FastAPI returns the prediction to the Streamlit frontend.

### Step 7 — Frontend displays the result

The user sees the predicted heart disease status.

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Machine Learning

* Scikit-learn
* XGBoost

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Frontend

* Streamlit

## Model Serialization

* Joblib / Pickle

## Deployment

* Render
* Streamlit Cloud

## Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/ashutoshkashyap04/Heart-disease-prediction.git
```

Navigate into the project:

```bash
cd Heart-disease-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Backend Locally

From the project root:

```bash
uvicorn backend.app:app --reload
```

The FastAPI application will run locally.

By default:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

# ▶️ Running the Frontend Locally

Run:

```bash
streamlit run frontend/app.py
```

The Streamlit application will open in the browser.

Make sure the frontend is configured to communicate with the correct FastAPI backend URL.

For local development, this would typically be:

```text
http://127.0.0.1:8000/predict
```

For production, the frontend uses the deployed Render API.

---

# 🧪 Model Development Workflow

The project follows a structured experimental workflow.

### Phase 1 — Data Preparation

```text
Raw Dataset
     ↓
Cleaning
     ↓
EDA
     ↓
Train/Test Split
```

### Phase 2 — Preprocessing

```text
Categorical Features
       ↓
One-Hot Encoding

Numerical Features
       ↓
StandardScaler

       ↓

Processed Train/Test Data
```

### Phase 3 — Model Selection

```text
9 Classification Models
          ↓
      Evaluation
          ↓
      F1 Ranking
          ↓
      Top 6 Models
```

### Phase 4 — Optimization

```text
Top 6 Models
     ↓
Hyperparameter Tuning
     ↓
F1 Score Comparison
     ↓
Top 3 Models
```

### Phase 5 — Ensemble Learning

```text
Random Forest
      +
     SVM
      +
   AdaBoost
      ↓
Stacking Classifier
      ↓
Logistic Regression
      ↓
Final Prediction
```

### Phase 6 — Deployment

```text
Trained Model
      ↓
Serialization
      ↓
FastAPI Backend
      ↓
Render
      ↓
Streamlit Frontend
      ↓
Streamlit Cloud
```

---

# 📁 Role of Important Files

| File                                     | Purpose                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `heart.csv`                              | Original raw dataset                                  |
| `raw/train.csv`                          | Training data after initial data preparation          |
| `raw/test.csv`                           | Testing data after initial data preparation           |
| `processed/train.csv`                    | Preprocessed training dataset                         |
| `processed/test.csv`                     | Preprocessed testing dataset                          |
| `columns.pkl`                            | Stores the expected model feature columns             |
| `scaler.pkl`                             | Stores the fitted StandardScaler                      |
| `stacking_classifier_model.pkl`          | Serialized final Stacking Classifier                  |
| `backend/app.py`                         | FastAPI application and prediction logic              |
| `backend/schemas.py`                     | Request validation schemas                            |
| `frontend/app.py`                        | Streamlit user interface                              |
| `01_data_loading_and_eda.ipynb`          | Data loading, cleaning, EDA and dataset splitting     |
| `02_data_preprocessing.ipynb`            | Encoding, scaling and preprocessing artifact creation |
| `03_model_training_and_evaluation.ipynb` | Model training, evaluation, tuning and stacking       |

---

# 🔐 Why Save the Preprocessing Objects?

A machine learning model expects data in exactly the same format used during training.

For example, during training:

```text
Age
RestingBP
Cholesterol
MaxHR
Oldpeak
...
```

were transformed using the fitted preprocessing pipeline.

When a new user enters data, the backend must perform the **same transformations**.

Therefore:

```text
Training Data
      ↓
Fit Encoder / Scaler
      ↓
Train Model
```

and later:

```text
New User Data
      ↓
Same Feature Transformation
      ↓
Same Scaler
      ↓
Model
      ↓
Prediction
```

The saved `scaler.pkl` and `columns.pkl` help maintain consistency between training and inference.

---

# 📌 Important Design Decision

The frontend does **not** ask the user to manually enter one-hot encoded variables.

Instead, the user provides the original human-readable values:

```text
Sex = M
ChestPainType = ATA
RestingECG = Normal
ExerciseAngina = N
ST_Slope = Up
```

The backend handles the transformation required by the machine learning model.

This results in a cleaner user experience and keeps preprocessing logic close to the prediction service.

---

# 📈 Key Results

The final Stacking Classifier achieved:

| Metric     |       Score |
| ---------- | ----------: |
| Accuracy   |    **0.88** |
| ROC-AUC    | **0.93675** |
| Class 0 F1 |    **0.87** |
| Class 1 F1 |    **0.89** |

The model demonstrated strong classification performance on the held-out evaluation dataset.

---

# 🧠 Key Machine Learning Concepts Demonstrated

This project demonstrates practical implementation of:

* Exploratory Data Analysis
* Train/Test Splitting
* Categorical Encoding
* One-Hot Encoding
* Feature Standardization
* Model Comparison
* Classification Metrics
* Precision
* Recall
* F1 Score
* ROC-AUC
* Hyperparameter Tuning
* Ensemble Learning
* Stacking
* Model Serialization
* Inference Preprocessing
* REST API Development
* FastAPI
* Pydantic Validation
* Frontend/API Integration
* Cloud Deployment

---

# ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**.

The prediction generated by this application should **not** be considered a medical diagnosis or a substitute for professional medical advice.

Real-world clinical decision-making requires qualified healthcare professionals, validated clinical systems, appropriate medical datasets, and extensive clinical validation.

---

# 🔮 Future Improvements

Potential improvements to the project include:

* Implementing a unified preprocessing pipeline using Scikit-learn `Pipeline` and `ColumnTransformer`.
* Adding automated model retraining workflows.
* Adding model explainability using SHAP or similar techniques.
* Adding probability/confidence visualization.
* Adding comprehensive API documentation.
* Adding automated testing for backend endpoints.
* Adding CI/CD using GitHub Actions.
* Containerizing the backend using Docker.
* Adding logging and monitoring.
* Implementing input validation and better error handling.
* Adding model versioning.
* Improving the frontend user experience.
* Adding a dedicated model-performance dashboard.

---

# 👨‍💻 Author

**Ashutosh Kashyap**

BS-MS in Artificial Intelligence and Cyber Security
IIT Patna

### GitHub

`https://github.com/ashutoshkashyap04`

### Project Repository

`https://github.com/ashutoshkashyap04/Heart-disease-prediction`

---

# ⭐ If You Found This Project Useful

If you found this project interesting or useful, consider giving the repository a ⭐ on GitHub.

---


