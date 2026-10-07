"""
Data Preprocessing and Feature Pipeline for Calorie Prediction Models
Handles feature transformation, scaling, categorical encoding, and inference formatting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERICAL_COLS = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
CATEGORICAL_COLS = ['Gender', 'Exercise_Type', 'Activity_Level']
TARGET_COL = 'Calories'

EXERCISE_TYPES = [
    "Walking", "Running", "Cycling", "Swimming", 
    "Gym", "HIIT", "Yoga", "Strength Training", "Other"
]

ACTIVITY_LEVELS = [
    "Sedentary", "Lightly Active", "Moderately Active", 
    "Very Active", "Extremely Active"
]

GENDERS = ["male", "female"]

def load_and_preprocess_data(data_path="data/calories.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, performs one-hot encoding for categorical variables,
    scales numerical features, and returns train/test splits.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Drop identifier if present
    if "User_ID" in df.columns:
        df = df.drop(columns=["User_ID"])
        
    # Clean and fill any missing values
    df = df.dropna()
    
    # Ensure standard categorical types
    df['Gender'] = df['Gender'].str.lower()
    
    # One-hot encode categoricals with consistent categories
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]
    
    feature_names = list(X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Fit scaler on numerical columns only
    scaler.fit(X_train[NUMERICAL_COLS])
    
    # Transform train & test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[NUMERICAL_COLS] = scaler.transform(X_train[NUMERICAL_COLS])
    X_test_scaled[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    metadata = {
        "feature_names": feature_names,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "exercise_types": EXERCISE_TYPES,
        "activity_levels": ACTIVITY_LEVELS,
        "genders": GENDERS
    }
    joblib.dump(metadata, "models/feature_info.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def transform_single_input(user_input_dict, models_dir="models", return_df=False):
    """
    Transforms a single user dictionary input into the scaled feature array or DataFrame
    ready for model inference.
    
    user_input_dict format:
    {
        'Age': 25,
        'Gender': 'male',
        'Height': 175.0,
        'Weight': 70.0,
        'Duration': 30,
        'Heart_Rate': 130,
        'Body_Temp': 38.5,
        'Exercise_Type': 'Running',
        'Activity_Level': 'Moderately Active'
    }
    """
    feature_info_path = os.path.join(models_dir, "feature_info.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    
    if not os.path.exists(feature_info_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Preprocessor artifacts not found. Please train models first.")
        
    feature_info = joblib.load(feature_info_path)
    scaler = joblib.load(scaler_path)
    feature_names = feature_info["feature_names"]
    
    # Create single row DataFrame
    row_df = pd.DataFrame([user_input_dict])
    row_df['Gender'] = row_df['Gender'].str.lower()
    
    # One-hot encode matching the categories
    row_encoded = pd.get_dummies(row_df, columns=feature_info["categorical_cols"], drop_first=False)
    
    # Ensure all training feature columns exist, fill missing dummy columns with 0
    full_row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for col in row_encoded.columns:
        if col in full_row.columns:
            full_row[col] = row_encoded[col].values
            
    # Scale numerical columns
    full_row[NUMERICAL_COLS] = scaler.transform(full_row[NUMERICAL_COLS])
    
    if return_df:
        return full_row
    return full_row.values.astype(np.float32)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    print(f"Features count: {len(features)}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
