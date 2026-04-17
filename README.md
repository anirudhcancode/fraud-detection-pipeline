# Fraud Detection Pipeline & ML System

An end-to-end fraud detection system built with PySpark, PostgreSQL, scikit-learn, XGBoost, and FastAPI.

## Architecture

## Project Structure
## Tech Stack

- **Data Processing:** PySpark, pandas
- **Database:** PostgreSQL (Docker)
- **ML Models:** Random Forest, XGBoost (scikit-learn)
- **API:** FastAPI, Uvicorn
- **Infra:** Docker, Git

## Pipeline Phases

### Phase 1 — Data Ingestion
- Loads Kaggle credit card fraud dataset (284,807 rows)
- Ingests raw CSV into PostgreSQL using SQLAlchemy

### Phase 2 — ETL & Feature Engineering
- PySpark job reads from PostgreSQL and transforms data
- Engineered features:
  - `amount_log` — log-normalized transaction amount
  - `hour` — transaction hour extracted from time
  - `tx_count_rolling` — rolling transaction count (window=100)
  - `avg_amount_rolling` — rolling average transaction amount
  - `high_value_flag` — flags transactions above $1,000
- Output written to Parquet format

### Phase 3 — Model Training
- Trained Random Forest and XGBoost classifiers
- Handled class imbalance (0.17% fraud rate) using class_weight="balanced"
- Evaluation results:

| Model | ROC-AUC | Fraud Precision | Fraud Recall |
|---|---|---|---|
| Random Forest | 0.9839 | 0.77 | 0.83 |
| XGBoost | 0.9827 | 0.89 | 0.84 |

### Phase 4 — FastAPI Serving
- REST API exposes model predictions
- POST /predict accepts transaction features
- Returns fraud label and probability score

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker Desktop
- Java 17

### Run PostgreSQL
docker compose up -d

### Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### Run the pipeline
python src/ingest.py
python src/transform.py
python src/train.py

### Start the API
uvicorn api.main:app --reload

### Test the API
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"features": [0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, -0.08, -0.07, -0.27, -0.16, -0.15, -0.06, -0.08, -0.03, 0.08, -0.41, 0.08, 0.08, -0.19, -1.26, 0.29, 0.48, -0.21, 0.03, 0.02, 150.0, 4.61, 0, 0]}'

## Dataset

Kaggle Credit Card Fraud Detection — 284,807 transactions, 492 fraud cases (0.17%)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud