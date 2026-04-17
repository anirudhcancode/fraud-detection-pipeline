import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a transaction is fraudulent",
    version="1.0.0"
)

# Load model at startup
model = joblib.load("models/random_forest.pkl")

# Define input schema
class Transaction(BaseModel):
    features: List[float]

# Health check endpoint
@app.get("/")
def root():
    return {"status": "Fraud Detection API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Fraud" if prediction == 1 else "Legit",
        "fraud_probability": round(float(probability), 4)
    }