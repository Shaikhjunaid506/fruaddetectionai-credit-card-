import time
import random
import requests
from faker import Faker

fake = Faker()

# API Endpoint
API_URL = "http://localhost:8000/predict_fraud"

# Generate a properly formatted transaction matching the API's expected fields
def generate_transaction():
    return {
        "Transaction_ID": str(random.randint(1000, 9999)),
        "User_ID": random.randint(10000, 99999),
        "Amount": round(random.uniform(1, 10000), 2),
        "Transaction_Hour": random.randint(0, 23),
        "Transaction_Day": random.randint(1, 31),
        "Transaction_Type": random.choice(["Online", "POS", "ATM", "Transfer"]),
        "Merchant_Category": random.choice(["Electronics", "Gambling", "Retail", "Travel", "Crypto"]),
        "Card_Type": random.choice(["Credit", "Debit", "Prepaid"]),
        "Location": random.choice(["USA", "UK", "India", "Germany", "China"]),
        "User_Risk_Score": round(random.uniform(0, 1), 2),
        "Previous_Fraud_Flag": random.choice([0, 1]),
        "Account_Age_Days": random.randint(30, 5000)
    }

# Simulate transactions and send them to the API
def simulate_transactions(num_transactions=50, delay=1):
    for _ in range(num_transactions):
        transaction = generate_transaction()
        response = requests.post(API_URL, json=transaction)

        # Print transaction & response for debugging
        print(f"🔍 Transaction Sent: {transaction}")
        try:
            print(f"📢 API Response: {response.status_code} - {response.json()}\n")
        except requests.exceptions.JSONDecodeError:
            print(f"❌ Error: Invalid response from API: {response.text}")

        time.sleep(delay)  # Add delay between transactions

# Start simulation
if __name__ == "__main__":
    simulate_transactions(num_transactions=50, delay=1)
