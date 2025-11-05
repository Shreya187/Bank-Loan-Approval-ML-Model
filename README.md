# 🏦 Bank Loan Approval Prediction

## 📘 Overview
This project predicts whether a loan application will be **approved or not approved** based on applicant information such as income, gender, marital status, dependents, credit history, and property area.

It helps banks and financial institutions make **data-driven decisions** faster.

---

## 🤖 Machine Learning Models Used
1. **Logistic Regression**
   - Used for binary classification (Yes/No).
   - Provides probabilistic output and works well for linearly separable data.

2. **Decision Tree Classifier**
   - Splits data into multiple branches based on feature conditions.
   - Captures non-linear patterns and provides visual interpretability.

---

## 📂 Files in this Repository
| File | Description |
|------|--------------|
| `loan_prediction.py` | Main Python program containing all preprocessing, visualization, model training, and evaluation steps. |
| `train.csv` | Dataset used for training and testing the model. |
| `README.md` | Explanation and documentation for the project. |

---

## 🧠 How It Works
1. Loads the dataset (`train.csv`) using **Pandas**.  
2. Handles missing data and encodes categorical features.  
3. Splits data into **training and testing sets**.  
4. Trains both **Logistic Regression** and **Decision Tree** models.  
5. Compares model performance using **accuracy score**, **confusion matrix**, and **classification report**.  
6. Displays visualizations (income vs approval, education vs approval, etc.).  
7. Predicts whether a loan will be approved for new input data.

---

## 📈 Results
Both models achieved high accuracy, with **Credit History** and **Applicant Income** being the most important predictors of loan approval.

---

## 👩‍💻 Author
**Shreya Chakraborty**

---

## 🏁 Usage
To run this project on your system:
```bash
python3 loan_prediction.py
# Bank-Loan-Approval-ML-Model
Machine Learning model to predict bank loan approval using Logistic Regression and Decision Tree.
