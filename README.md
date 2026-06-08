# Titanic Survival Prediction 🚢

A machine learning project built on the classic [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) dataset. The goal is to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, and fare.

---

## 📌 Project Overview

This notebook walks through a full ML pipeline — from raw data preprocessing to generating a Kaggle submission file — using two classification models: Logistic Regression and Random Forest.

---

## 📂 Dataset

- **train.csv** — Training data with survival labels
- **test.csv** — Test data for generating predictions
- Source: [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)

---



 1. Data Preprocessing
- Encoded categorical features: `Sex` (male → 0, female → 1) and `Embarked` (C → 0, Q → 1, S → 2)
- Handled missing values in `Age` using **mean imputation** (Scikit-learn `SimpleImputer`)
- Dropped irrelevant columns: `Name`, `Ticket`, `Cabin`, `Embarked`, `PassengerId`

 2. Exploratory Data Analysis (EDA)
Visualized survival rates across multiple features using Matplotlib and Seaborn:
- Survival by Passenger Class (`Pclass`)
- Survival by Number of Siblings/Spouses (`SibSp`)
- Survival by Number of Parents/Children (`Parch`)
- Fare distribution by survival status

 3. Feature Scaling
- Applied **StandardScaler** to normalize `Age` and `Fare` columns

 4. Model Training
Two models were trained and compared:

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of 100 decision trees (entropy criterion) |
5. Model Evaluation
- Evaluated using **Confusion Matrix** and **Accuracy Score**
- Applied **5-Fold Cross Validation** on the Random Forest model to get a reliable accuracy estimate
 6. Prediction & Submission
- Applied the same preprocessing pipeline to the test set
- Generated a `submission.csv` file for Kaggle submission

---

Libraries Used

- `NumPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`
- `Scikit-learn`

--- Results

| Model | Cross-Validation Accuracy |
|---|---|
| Random Forest (5-fold CV) | ~82–84% |

---

How to Run

1. Clone this repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/data) and place `train.csv` and `test.csv` in the project folder
3. Open `Titanic_.ipynb` in Jupyter Notebook or VS Code
4. Run all cells in order

---

Author

**Yahia Mohtady**
GitHub: [YahiaZeada06](https://github.com/YahiaZeada06)