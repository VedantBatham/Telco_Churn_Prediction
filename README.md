# Telco Customer Churn Prediction

This repository contains a Machine Learning project for predicting customer churn in the telecommunications industry using the **Telco Customer Churn dataset** from Kaggle. The project leverages **XGBoost**, **scaling**, **PCA**, and **GridSearchCV** for hyperparameter tuning.

---

## Dataset

- **Source:** [Telco Customer Churn Dataset - Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)  
- **Records:** 7043 customer records  
- **Features:** 21 features including demographic info, account info, and services subscribed  
- **Target Variable:** `Churn` (1 = customer churned, 0 = customer retained)  

**Key Features Include:**  
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`  
- `tenure`, `MonthlyCharges`, `TotalCharges`  
- `Contract`, `PaymentMethod`, `InternetService`, `TechSupport`, etc.

---

## Project Steps

### 1. Data Cleaning
- Converted `TotalCharges` to numeric and filled missing values.  
- Dropped `customerID` column.

### 2. Encoding Categorical Features
- Binary columns encoded with `LabelEncoder`.  
- Multi-category columns encoded with `One-Hot Encoding`.

### 3. Handling Imbalanced Classes
- SMOTE used to oversample the minority class (churned customers).

### 4. Train-Test Split, Scaling & PCA
- Split dataset 80:20 for training and testing.  
- Scaled features using `StandardScaler`.  
- Applied **PCA** for dimensionality reduction.

### 5. Model Training
- XGBoost Classifier trained with **GridSearchCV** to find the best hyperparameters.  
- Tuned parameters:  
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

### 6. Model Evaluation
- Metrics used:  
  - ROC-AUC score  
  - Confusion matrix  
  - Classification report

### 7. Model Saving
- Final model saved using `pickle` for later use.

---

## Results

- **Best ROC-AUC (GridSearchCV):** ~0.902  
- **Best Parameters Found:**  
  ```text
  {'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 5,
   'n_estimators': 200, 'subsample': 0.8}

# Usage
- Clone the repository:
git clone <your-repo-url>
cd telco-churn-prediction

- Install dependencies:
pip install -r requirements.txt

- Run the notebook:
jupyter notebook Telco_Churn_Prediction.ipynb

- Load the trained model for prediction:
import pickle

with open('xgb_telco_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_new)

