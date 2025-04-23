import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load Data
df = pd.read_csv("D:/Credit-Card-Fraud-Detection-Using-Machine-Learning-main/Credit Card Fraud Detection Using Machine Learning/Code/enhanced_creditcard.csv")

# Ensure no missing values
print("\n🔍 **Data Quality Check:**")
print(df.isnull().sum())

# Check Class Imbalance
fraud_cases = df[df["Class"] == 1].shape[0]
non_fraud_cases = df[df["Class"] == 0].shape[0]
fraud_ratio = fraud_cases / (fraud_cases + non_fraud_cases) * 100
print(f"\n⚠️ **Fraud Case Ratio:** {fraud_ratio:.2f}% fraud cases detected.")

# --- 🔹 Statistical Justification for Fraud Trends ---
# Fraud Occurrence by Transaction Time
fraud_hours = df[df['Class'] == 1]['Transaction_Hour']
non_fraud_hours = df[df['Class'] == 0]['Transaction_Hour']

# T-Test to Check Statistical Difference
t_stat, p_val = ttest_ind(fraud_hours, non_fraud_hours)
print("\n📊 **T-Test for Fraud Occurrence by Hour:**")
print(f"T-Statistic = {t_stat:.3f}, p-value = {p_val:.10f}")
if p_val < 0.05:
    print("✅ Fraud significantly varies by time of day.")
else:
    print("❌ No significant time-based fraud pattern found.")

# --- 🔹 Prepare Data for Machine Learning ---
X = df.drop(columns=['Class'])
y = df['Class']

# Handle Categorical Features
non_numeric_cols = ['Associate_Name', 'Education_Status', 'Job_Title']

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Convert categories to numbers
    label_encoders[col] = le  # Store encoder for future use

# Drop Associate_Name before applying SMOTE, as it's just a name
X = X.drop(columns=['Associate_Name'])

# Handle Imbalanced Data
smote = SMOTE(sampling_strategy=0.05, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 🔹 Train Latest Fraud Detection Models ---
print("\n🚀 Training XGBoost...")
xgb_model = xgb.XGBClassifier(scale_pos_weight=99, subsample=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

print("\n🔥 Training CatBoost...")
cat_model = cb.CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=0)
cat_model.fit(X_train_scaled, y_train)

print("\n⚡ Training LightGBM...")
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train_scaled, y_train)

print("\n🌳 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- 🔹 Predict & Evaluate Models ---
models = {
    "XGBoost": xgb_model,
    "CatBoost": cat_model,
    "LightGBM": lgb_model,
    "Random Forest": rf_model
}

# Store model performance in a table
performance_table = []

for name, model in models.items():
    print(f"\n📌 **Evaluating {name}:**")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    
    performance_table.append([name, accuracy, precision, recall, f1, roc_score])

    # Classification Report
    print("\n📊 **Classification Report:**")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"📊 Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, marker='.')
    plt.title(f"🎯 Precision-Recall Curve - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.show()

    print(f"\n🔍 **ROC-AUC Score ({name}):** {roc_score:.4f}")

# Convert performance metrics into a DataFrame for easy display
performance_df = pd.DataFrame(performance_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC Score"])

# Display the model performance table
print("\n📌 **Model Performance Summary:**")
print(performance_df)

# --- 🔹 Justifying SHAP Feature Importance ---
print("\n📌 **SHAP Explanation: Why a Transaction is Fraud?**")
shap.initjs()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled[:500])
shap.summary_plot(shap_values, X_test_scaled[:500])
