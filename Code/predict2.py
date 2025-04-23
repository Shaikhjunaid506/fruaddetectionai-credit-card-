import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
import shap

# Load the enhanced dataset
df = pd.read_csv("D:/Credit-Card-Fraud-Detection-Using-Machine-Learning-main/Credit Card Fraud Detection Using Machine Learning/Code/enhanced_creditcard.csv")
df = df.dropna()  # Remove missing values

# --- Fraud Risk Analysis by Age Group ---
df['Age_Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 80], labels=['18-30', '30-45', '45-60', '60+'])
fraud_by_age = df[df['Class'] == 1]['Age_Group'].value_counts()

# --- Fraud Analysis by Job Title ---
fraud_by_job = df[df['Class'] == 1]['Job_Title'].value_counts()

# --- Fraud Detection Effectiveness ---
noticed_fraud_rate = df['Noticed_Fraud_Transactions'].sum() / df['Frauded_Count'].sum() * 100
unnoticed_fraud_rate = 100 - noticed_fraud_rate

# --- Fraud by Education Level ---
fraud_by_education = df[df['Class'] == 1]['Education_Status'].value_counts()

# --- High-Risk Users (Based on Risk Score) ---
high_risk_users = df[df['Risk_Score'] > df['Risk_Score'].quantile(0.9)][['Associate_Name', 'Risk_Score', 'Frauded_Count']].sort_values(by='Risk_Score', ascending=False).head(10)

# --- Fraud by Transaction Type & Merchant ---
fraud_by_transaction_type = df[df['Class'] == 1].filter(like='Transaction_Type_').sum()
fraud_by_merchant = df[df['Class'] == 1].filter(like='Merchant_Category_').sum()

# --- Geographic Risk Analysis ---
fraud_by_country = df[df['Class'] == 1].filter(like='Location_').sum()

# --- Predicting Future Fraud Risk (Using Transaction_Day Instead of Transaction_Month) ---
fraud_increase_rate = df[df['Class'] == 1]['Transaction_Day'].value_counts().pct_change().mean() * 100
future_fraud_risk = len(df[df['Class'] == 1]) * (1 + (fraud_increase_rate / 100))

# --- Visualizing Fraud Trends ---
plt.figure(figsize=(10, 5))
sns.barplot(x=fraud_by_age.index, y=fraud_by_age.values, palette='coolwarm')
plt.xlabel("Age Group")
plt.ylabel("Fraud Cases")
plt.title("Fraud Distribution by Age Group")
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(x=fraud_by_job.index[:10], y=fraud_by_job.values[:10], palette='muted')
plt.xticks(rotation=45)
plt.xlabel("Job Title")
plt.ylabel("Fraud Cases")
plt.title("Top 10 Fraud-Prone Job Titles")
plt.show()

# --- Displaying Fraud Insights ---
print("\n✅ **1. Fraud Risk by Age & Job Title**")
print(fraud_by_age)
print("\n✅ **2. Fraud Risk by Education Level**")
print(fraud_by_education)
print("\n✅ **3. Fraud Detection Effectiveness**")
print(f"✅ **Noticed Fraud Rate:** {noticed_fraud_rate:.2f}% | ❌ **Unnoticed Fraud Rate:** {unnoticed_fraud_rate:.2f}%")
print("\n✅ **4. High-Risk Users**")
print(high_risk_users)

print("\n✅ **5. Fraud by Transaction Type & Merchant**")
print("💳 **Transaction Types Most Targeted:**")
print(fraud_by_transaction_type)
print("🏪 **Fraud-Prone Merchant Categories:**")
print(fraud_by_merchant)

print("\n✅ **6. Geographic Fraud Risk**")
print(fraud_by_country)

print("\n✅ **7. Future Fraud Prediction**")
print(f"📈 **Predicted Fraud Cases Next Month:** ~{int(future_fraud_risk)} cases expected.")
print(f"📉 **Fraud Growth Rate:** {fraud_increase_rate:.2f}%")
