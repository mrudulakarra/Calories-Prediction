"""
Deep Learning Training Pipeline
Trains a Dense Neural Network (DNN) with TensorFlow / Keras for Calorie Prediction.
Evaluates metrics (MAE, MSE, RMSE, R2) and updates model_metadata.json.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import load_and_preprocess_data

def build_dnn_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.003),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

def train_dl_model(epochs=60, batch_size=64):
    print("Loading preprocessed dataset for Deep Learning...")
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    
    input_dim = X_train.shape[1]
    print(f"Building DNN model with input dimension: {input_dim}")
    model = build_dnn_model(input_dim)
    model.summary()
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    )
    
    print("\n--- Training Deep Neural Network ---")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    y_pred = model.predict(X_test).flatten()
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    print("\n[Deep Neural Network (DNN)]")
    print(f"  MAE : {mae:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²  : {r2:.4f}")
    
    dl_metrics = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }
    
    # Save model
    os.makedirs("models", exist_ok=True)
    dnn_save_path = "models/calories_dnn.keras"
    model.save(dnn_save_path)
    print(f"DNN model saved to {dnn_save_path}")
    
    # Update model_metadata.json
    metadata_path = "models/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"models": {}, "best_model": ""}
        
    metadata["models"]["Deep Neural Network (DNN)"] = dl_metrics
    
    # Find overall best model (highest R2)
    best_m = max(metadata["models"].items(), key=lambda x: x[1]["r2"])[0]
    metadata["best_model"] = best_m
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Updated metadata saved to {metadata_path}")
    print(f"Overall Best Model: {best_m} with R² = {metadata['models'][best_m]['r2']}")

if __name__ == "__main__":
    train_dl_model()
