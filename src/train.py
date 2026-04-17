import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

# Load processed parquet data
def load_data(parquet_path: str) -> pd.DataFrame:
    print(f"Loading processed data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def prepare_features(df: pd.DataFrame):
    # Drop non-feature columns
    drop_cols = ["Time", "Amount"]
    feature_cols = [c for c in df.columns if c not in drop_cols + ["label"]]

    X = df[feature_cols]
    y = df["label"]

    print(f"Features: {len(feature_cols)} columns")
    print(f"Fraud cases: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")
    return X, y

def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test, name: str):
    print(f"\n--- {name} Results ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    return auc

def save_model(model, name: str):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    df = load_data("data/processed")
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_auc = evaluate(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, "random_forest")

    # Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_auc = evaluate(xgb_model, X_test, y_test, "XGBoost")
    save_model(xgb_model, "xgboost")

    # Pick best model
    best = "random_forest" if rf_auc >= xgb_auc else "xgboost"
    print(f"\nBest model: {best} (AUC: {max(rf_auc, xgb_auc):.4f})")
    print("Training complete!")