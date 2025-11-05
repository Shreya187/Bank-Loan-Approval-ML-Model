# --- Task 1: Exploratory Data Analysis (EDA) ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
# Load dataset
data = pd.read_csv("train.csv")

# Display basic info
print("✅ Dataset Loaded Successfully!")
print("Shape of data:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Check missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# --- Visualization 1: Education vs Loan Status ---
plt.figure(figsize=(6,4))
sns.countplot(x="Education", hue="Loan_Status", data=data)
plt.title("Education vs Loan Approval Status")
plt.show()

# --- Visualization 2: Applicant Income Distribution ---
plt.figure(figsize=(6,4))
sns.histplot(data["ApplicantIncome"], kde=True, color="purple")
plt.title("Applicant Income Distribution")
plt.show()

# Summary of findings
print("""
📊 EDA Summary:
1. Graduates have a higher loan approval rate than non-graduates.
2. Most applicants have moderate income levels.
3. Some missing data exists in columns like Gender, LoanAmount, etc.
""")

# --- Task 2: Data Preprocessing ---
# Fill missing values
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)

# Encode categorical variables
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
le = LabelEncoder()
for col in cols:
    data[col] = le.fit_transform(data[col])

print("\n✅ After Encoding and Filling Missing Values:")
print(data.head())

# Separate features and target
X = data.drop(columns=['Loan_ID', 'Loan_Status'])
y = data['Loan_Status']

print("\nFeature Matrix (X) Shape:", X.shape)
print("Target Vector (y) Shape:", y.shape)

# --- Task 3: Model Building ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Model 2: Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

print("\n✅ Models Trained Successfully!")

# --- Task 4: Model Evaluation ---
y_pred_log = log_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
acc_tree = accuracy_score(y_test, y_pred_tree)

print("\n📊 Model Performance:")
print("Logistic Regression Accuracy:", round(acc_log * 100, 2), "%")
print("Decision Tree Accuracy:", round(acc_tree * 100, 2), "%")

print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, y_pred_log))

print("\n--- Decision Tree Report ---")
print(classification_report(y_test, y_pred_tree))

print("\nConfusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_log))
print("\nConfusion Matrix (Decision Tree):\n", confusion_matrix(y_test, y_pred_tree))

print("""
✅ Metric Used: Accuracy
Reason: For loan approval prediction, accuracy shows how often the model predicts correctly.
If the dataset had been highly unbalanced, F1-score would have been preferred instead.
""")
