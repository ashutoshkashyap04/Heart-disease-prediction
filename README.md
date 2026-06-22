# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project predicts the likelihood of heart disease using Machine Learning models trained on patient health data. The goal is to assist in early risk assessment by analyzing various medical attributes and providing a prediction along with the probability of heart disease.

The project follows a complete Machine Learning workflow including data preprocessing, feature engineering, model selection, evaluation, model saving, and deployment using Streamlit.

---

## 🚀 Features

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature encoding and transformation
* Training multiple Machine Learning models
* Model comparison using evaluation metrics
* ROC-AUC based performance analysis
* Probability prediction for risk assessment
* Interactive Streamlit web application
* Deployment on Streamlit Cloud

---

## 📊 Dataset

The dataset contains medical information such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise Induced Angina
* Oldpeak
* ST Slope
* Heart Disease (Target Variable)

Target Variable:

* 0 → No Heart Disease
* 1 → Heart Disease Present

---

## 🛠️ Machine Learning Workflow

### 1. Data Preprocessing

* Handled categorical features
* Applied one-hot encoding
* Ensured feature consistency
* Prepared data for model training

### 2. Exploratory Data Analysis

* Studied feature distributions
* Examined relationships between variables
* Identified important patterns in the dataset

### 3. Model Training

The following models were trained and evaluated:

* Logistic Regression
* K-Nearest Neighbor
* Random Forest Classifier
* Gradient Boosting Classifier
* AdaBoost Classifier
* Support Vector Machine (SVM)
* Gaussian Naive Bayes
* Decision Tree
* XG Boost
* Stacking Classifier

### 4. Model Evaluation

Models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

### 5. Best Model Selection

The final model was selected based on overall performance, particularly:

* F1 Score
* ROC-AUC Score

The chosen model achieved a ROC-AUC score of approximately **0.936**.

### 6. Model Saving

The trained model was saved using:

```python
joblib.dump(model, "heart_model.pkl")
```

---

## 🌐 Streamlit Application

The web application allows users to:

1. Enter patient health information.
2. Submit data through an intuitive interface.
3. Receive a prediction:

   * Low Risk
   * High Risk
4. View prediction probability.

Example Output:

```text
Prediction: Low Risk

Probability of Heart Disease: 5.01%
```


---

## 📦 Requirements

Main libraries used:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 📈 Evaluation Metrics

The project focuses on:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

These metrics provide a balanced evaluation of classification performance, especially for medical prediction tasks.

---



## 🎯 Learning Outcomes

Through this project, I learned:

* Data preprocessing techniques
* Exploratory Data Analysis
* Feature engineering
* Classification algorithms
* Model evaluation and comparison
* ROC-AUC analysis
* Model deployment using Streamlit
* End-to-end Machine Learning workflow

---

## 👨‍💻 Author

Ashutosh Kashyap

```

