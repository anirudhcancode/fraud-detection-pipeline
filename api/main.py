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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, -0.08, -0.07, -0.27, -0.16, -0.15, -0.06, -0.08, -0.03, 0.08, -0.41, 0.08, 0.08, -0.19, -1.26, 0.29, 0.48, -0.21, 0.03, 0.02, 150.0, 4.61, 0, 0]
                }
            ]
        }
    }

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