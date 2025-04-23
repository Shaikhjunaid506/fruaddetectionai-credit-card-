import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv("D:/Credit-Card-Fraud-Detection-Using-Machine-Learning-main/Credit Card Fraud Detection Using Machine Learning/Code/creditcard.csv")

# Check for missing values
df = df.dropna()

# Separate features and target variable
X = df.drop(columns=['Class'])  # Assuming 'Class' column has 0 (normal) and 1 (fraud)
y = df['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### XGBoost Model (Supervised Learning)
xgb_model = XGBClassifier(scale_pos_weight=99)  # Adjust weight for imbalanced data
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Model Performance:")
print(classification_report(y_test, y_pred_xgb))

### Isolation Forest (Unsupervised Anomaly Detection)
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
isolation_forest.fit(X_train)

# Predict anomalies (-1 = fraud, 1 = normal)
y_pred_iso = isolation_forest.predict(X_test)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]  # Convert -1 to 1 (fraud)

print("\nIsolation Forest Model Performance:")
print(classification_report(y_test, y_pred_iso))

# Identifying fraudulent transactions
df_test = pd.DataFrame(X_test, columns=df.drop(columns=['Class']).columns)
df_test['Actual_Class'] = y_test.values

df_test['XGBoost_Prediction'] = y_pred_xgb
df_test['IsolationForest_Prediction'] = y_pred_iso

# Display fraud transactions detected
fraud_cases = df_test[(df_test['XGBoost_Prediction'] == 1) | (df_test['IsolationForest_Prediction'] == 1)]
print("\nFraudulent Transactions Detected:")
print(fraud_cases)
