import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Disable TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations if causing issues

# Suppress other warnings globally
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("data/enhanced_creditcard.csv")

# Drop non-numeric columns that are not useful for training
df = df.drop(columns=["Associate_Name", "Education_Status", "Job_Title"])

# ✅ Generate 'User_Risk_Score'
df["User_Risk_Score"] = (
    df["Previous_Fraud_Flag"] * 75 +
    (100 - df["Account_Age_Days"] / df["Account_Age_Days"].max() * 100)
)
df["User_Risk_Score"] = df["User_Risk_Score"].clip(0, 150)

# Separate features and target
X = df.drop(columns=["Class"])  # Target = Class (fraud label)
y = df["Class"]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler & feature names
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
print("✅ Scaler & Feature Names Saved Successfully!")

# Handle imbalance using SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),  # Use Input layer for proper shape definition
    tf.keras.layers.LSTM(64, return_sequences=True),  # Reduced LSTM units for faster training
    tf.keras.layers.LSTM(32),  # Reduced number of units here as well
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


model.fit(X_train, y_train, epochs=1, batch_size=128, validation_data=(X_test, y_test))

# Save the model
model.save("models/fraud_lstm_model.h5")
print("✅ Model Training Completed & Saved Successfully!")
