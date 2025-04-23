import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import threading
from fastapi.responses import JSONResponse

# Load the trained model
model = tf.keras.models.load_model("models/fraud_lstm_model.h5")

# Load the scaler and feature names
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")  # Load expected feature names

app = FastAPI()

# Define the expected request structure
class FraudRequest(BaseModel):
    Transaction_ID: str
    User_ID: int
    Amount: float
    Transaction_Hour: int
    Transaction_Day: int
    Transaction_Type: str
    Merchant_Category: str
    Card_Type: str
    Location: str
    User_Risk_Score: float
    Previous_Fraud_Flag: int
    Account_Age_Days: int

# Function to preprocess incoming data
def preprocess_request(data: FraudRequest):
    """Convert incoming JSON request to model-compatible feature array."""
    # Convert dictionary to feature array in correct order
    input_features = {feature: 0 for feature in feature_names}  # Default all to 0
    
    # Update with actual values from request
    input_features.update({
        "Amount": data.Amount,
        "Transaction_Hour": data.Transaction_Hour,
        "Transaction_Day": data.Transaction_Day,
        "User_Risk_Score": data.User_Risk_Score,
        "Previous_Fraud_Flag": data.Previous_Fraud_Flag,
        "Account_Age_Days": data.Account_Age_Days,
        "Transaction_Velocity": data.Amount / (data.Transaction_Hour + 1),  # New feature
        "Risk_Score": data.User_Risk_Score * data.Previous_Fraud_Flag,  # New feature
    })

    # One-hot encode categorical fields
    if f"Transaction_Type_{data.Transaction_Type}" in input_features:
        input_features[f"Transaction_Type_{data.Transaction_Type}"] = 1
    if f"Merchant_Category_{data.Merchant_Category}" in input_features:
        input_features[f"Merchant_Category_{data.Merchant_Category}"] = 1
    if f"Card_Type_{data.Card_Type}" in input_features:
        input_features[f"Card_Type_{data.Card_Type}"] = 1
    if f"Location_{data.Location}" in input_features:
        input_features[f"Location_{data.Location}"] = 1

    # Convert dictionary to Pandas DataFrame
    input_df = pd.DataFrame([input_features], columns=feature_names)

    return input_df

# Store fraud logs globally
fraud_logs = []

@app.post("/predict_fraud")
def predict_fraud(data: FraudRequest):
    """API endpoint for fraud detection."""
    try:
        # Preprocess input data
        input_data = preprocess_request(data)
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)[0][0]
        fraud_risk = round(float(prediction), 4)

        # Log transaction details
        log_data = {"transaction": data.dict(), "fraud_risk": fraud_risk}
        fraud_logs.append(log_data)

        # Save logs to file
        with open("logs/fraud_predictions.json", "a") as log_file:
            log_file.write(json.dumps(log_data) + "\n")

        return {"fraud_risk": fraud_risk}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

@app.get("/fraud_logs", response_model=list)
def get_fraud_logs():
    """Return all fraud predictions from logs."""
    return JSONResponse(content=fraud_logs)

# Function to generate fraud insights every 2 minutes
def generate_fraud_insights():
    while True:
        if fraud_logs:
            df = pd.DataFrame([log["transaction"] for log in fraud_logs])
            df["fraud_risk"] = [log["fraud_risk"] for log in fraud_logs]

            # 1. Fraud Risk by Age & Job Title
            fraud_by_age = df.groupby("Account_Age_Days").size().sort_values(ascending=False).head(4)
            fraud_by_education = df.groupby("User_ID").size().sort_values(ascending=False).head(5)

            # 2. Fraud Detection Effectiveness
            noticed_fraud = df[df["fraud_risk"] >= 0.5].shape[0]
            unnoticed_fraud = df[df["fraud_risk"] < 0.5].shape[0]
            fraud_effectiveness = f"✅ **Noticed Fraud Rate:** {noticed_fraud / len(df):.2%} | ❌ **Unnoticed Fraud Rate:** {unnoticed_fraud / len(df):.2%}"

            # 3. High-Risk Users
            high_risk_users = df[df["fraud_risk"] >= 0.7].groupby("User_ID").size().reset_index(name="Fraud_Count").sort_values(by="Fraud_Count", ascending=False).head(10)

            # 4. Fraud by Transaction Type & Merchant
            fraud_by_transaction = df.groupby("Transaction_Type").size().sort_values(ascending=False).head(3)
            fraud_by_merchant = df.groupby("Merchant_Category").size().sort_values(ascending=False).head(4)

            # 5. Geographic Fraud Risk
            fraud_by_location = df.groupby("Location").size().sort_values(ascending=False).head(5)

            # 6. Future Fraud Prediction
            future_fraud_cases = len(df) * 1.2  # Assume fraud will grow by 20%
            fraud_growth_rate = "📈 **Predicted Fraud Cases Next Month:** ~ " + str(int(future_fraud_cases)) + " cases expected."

            # Display insights
            insights = f"""
✅ **1. Fraud Risk by Age & Job Title**  
{fraud_by_age.to_string()}

✅ **2. Fraud Risk by Education Level**  
{fraud_by_education.to_string()}

✅ **3. Fraud Detection Effectiveness**  
{fraud_effectiveness}

✅ **4. High-Risk Users**  
{high_risk_users.to_string()}

✅ **5. Fraud by Transaction Type & Merchant**  
💳 **Transaction Types Most Targeted:**  
{fraud_by_transaction.to_string()}

🏪 **Fraud-Prone Merchant Categories:**  
{fraud_by_merchant.to_string()}

✅ **6. Geographic Fraud Risk**  
{fraud_by_location.to_string()}

✅ **7. Future Fraud Prediction**  
{fraud_growth_rate}
"""
            print(insights)

        time.sleep(120)  # Run every 2 minutes

# Start fraud insights generation in the background
threading.Thread(target=generate_fraud_insights, daemon=True).start()
