"""
Machine Learning Training Pipeline
Trains Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.
Evaluates MAE, MSE, RMSE, and R2 metrics and saves models to models/ directory.
"""

import os
import sys
import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_and_preprocess_data

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    print(f"[{model_name}]")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²  : {r2:.4f}")
    
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

def train_ml_models():
    print("Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    
    os.makedirs("models", exist_ok=True)
    metadata_path = "models/model_metadata.json"
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"models": {}, "best_model": ""}
        
    # 1. Linear Regression
    print("\n--- Training Linear Regression ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr, X_test, y_test, "Linear Regression")
    joblib.dump(lr, "models/linear_regression.pkl")
    metadata["models"]["Linear Regression"] = lr_metrics
    
    # 2. Random Forest Regressor
    print("\n--- Training Random Forest Regressor ---")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
    joblib.dump(rf, "models/random_forest.pkl")
    metadata["models"]["Random Forest"] = rf_metrics
    
    # 3. Gradient Boosting Regressor
    print("\n--- Training Gradient Boosting Regressor ---")
    gb = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_metrics = evaluate_model(gb, X_test, y_test, "Gradient Boosting")
    joblib.dump(gb, "models/gradient_boosting.pkl")
    metadata["models"]["Gradient Boosting"] = gb_metrics
    
    # Determine best model among ML models (highest R2 or lowest RMSE)
    best_m = max(metadata["models"].items(), key=lambda x: x[1]["r2"])[0]
    metadata["best_model"] = best_m
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nML training complete! Metadata saved to {metadata_path}")
    print(f"Current best model: {best_m}")

if __name__ == "__main__":
    train_ml_models()
