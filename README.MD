# ❤️ Heart Disease Prediction using Machine Learning

A machine learning project that predicts the likelihood of heart disease based on patient health and clinical parameters. The project includes data preprocessing, multiple machine learning models, hyperparameter tuning, model saving, and a professional Streamlit web application.

## 🚀 Live Demo

🌐 **Streamlit Application:**  
[Click here to try the live application](https://patient-heart-disease-prediction.streamlit.app/)

📂 **GitHub Repository:**  
[View the source code on GitHub](https://github.com/nikhil-hanagal/heart-disease-prediction.git)

---

## 📌 Project Overview

Heart disease is one of the major health challenges worldwide. Early identification of potential heart disease can help support further medical evaluation.

In this project, machine learning techniques are used to classify patients into two categories:

- **0 → No heart disease**
- **1 → Heart disease**

The project uses patient information such as age, blood pressure, cholesterol, maximum heart rate, chest pain type, exercise-induced angina, and other clinical features.

The trained machine learning model is integrated into a Streamlit application where users can enter patient information and receive a prediction.

> ⚠️ **Disclaimer:** This application is developed for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## 🎯 Objectives

The main objectives of this project are:

- Perform exploratory data analysis on the heart disease dataset.
- Handle missing values effectively.
- Preprocess numerical and categorical features.
- Train multiple machine learning classification models.
- Compare model performance.
- Perform hyperparameter tuning.
- Save the best-performing model.
- Build an interactive Streamlit application.
- Deploy the application online.

---

## 📊 Dataset

The project uses the **UCI Heart Disease dataset**.

The dataset contains clinical information about patients and includes features such as:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex of the patient |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of the ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `num` | Original target variable |

The original target variable contains multiple classes.

For this project, it was converted into a binary classification problem:

```python
y = (df["num"] > 0).astype(int)