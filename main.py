from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Credit Card Fraud Detection API")

# Load trained model (update filename if different)
model = joblib.load("fraud_model.pkl")


class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def health_check():
    return {"status": "Fraud Detection API is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]

    return {
        "fraud": bool(prediction),
        "label": "Fraud" if prediction == 1 else "Normal"
    }
