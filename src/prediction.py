"""
Prediction Engine for Calorie Burn AI
Loads trained ML and DL models, performs inference, determines intensity zones,
computes goal contribution, and generates actionable AI recommendations.
"""

import os
import sys
import json
import joblib
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import transform_single_input

MODELS_DIR = "models"
_LOADED_MODELS = {}
_METADATA = None

def get_model_metadata():
    global _METADATA
    if _METADATA is None:
        meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {"models": {}, "best_model": "Random Forest"}
    return _METADATA

def load_model(model_name=None):
    """
    Loads and caches requested model:
    'Deep Neural Network (DNN)', 'Gradient Boosting', 'Random Forest', or 'Linear Regression'.
    If model_name is None, loads the best model.
    """
    meta = get_model_metadata()
    if model_name is None or model_name == "Auto (Best Model)":
        model_name = meta.get("best_model", "Deep Neural Network (DNN)")
        
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name], model_name
        
    if model_name == "Deep Neural Network (DNN)":
        dnn_path = os.path.join(MODELS_DIR, "calories_dnn.keras")
        if os.path.exists(dnn_path):
            model = tf.keras.models.load_model(dnn_path)
            _LOADED_MODELS[model_name] = model
            return model, model_name
        else:
            raise FileNotFoundError(f"DNN model not found at {dnn_path}")
            
    elif model_name == "Gradient Boosting":
        path = os.path.join(MODELS_DIR, "gradient_boosting.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            _LOADED_MODELS[model_name] = model
            return model, model_name
            
    elif model_name == "Random Forest":
        path = os.path.join(MODELS_DIR, "random_forest.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            _LOADED_MODELS[model_name] = model
            return model, model_name
            
    elif model_name == "Linear Regression":
        path = os.path.join(MODELS_DIR, "linear_regression.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            _LOADED_MODELS[model_name] = model
            return model, model_name
            
    # Fallback to any available
    for m_name, filename in [
        ("Gradient Boosting", "gradient_boosting.pkl"),
        ("Random Forest", "random_forest.pkl"),
        ("Linear Regression", "linear_regression.pkl")
    ]:
        p = os.path.join(MODELS_DIR, filename)
        if os.path.exists(p):
            model = joblib.load(p)
            _LOADED_MODELS[m_name] = model
            return model, m_name
            
    raise FileNotFoundError("No trained models found in models/ directory.")

def calculate_intensity_and_zone(heart_rate, age, duration):
    """
    Calculates workout intensity category and target heart rate zone.
    """
    max_hr = 220 - age
    pct_max = (heart_rate / max(max_hr, 100)) * 100
    
    if pct_max < 60:
        zone = "Warm Up / Light Recovery (Zone 1)"
        intensity = "Low"
        color = "#3B82F6" # blue
    elif pct_max < 70:
        zone = "Fat Burn (Zone 2)"
        intensity = "Moderate"
        color = "#10B981" # green
    elif pct_max < 85:
        zone = "Aerobic / Cardio (Zone 3)"
        intensity = "Vigorous"
        color = "#F59E0B" # amber
    else:
        zone = "Peak / Anaerobic (Zone 4/5)"
        intensity = "Maximum"
        color = "#EF4444" # red
        
    return {
        "intensity": intensity,
        "zone_name": zone,
        "pct_max_hr": round(pct_max, 1),
        "color": color,
        "max_hr": max_hr
    }

def generate_ai_recommendation(calories, duration, heart_rate, age, exercise_type, intensity):
    """
    Produces evidence-based AI fitness recommendations tailored to workout parameters.
    """
    cal_per_min = round(calories / max(duration, 1), 1)
    recs = []
    
    if intensity == "Maximum":
        recs.append("🔥 High-intensity effort detected! Prioritize hydration (500-750ml) and consume protein/carbs within 45 mins.")
    elif intensity == "Vigorous":
        recs.append("⚡ Excellent cardiovascular stimulus. Great zone for stamina and functional endurance.")
    elif intensity == "Moderate":
        recs.append("🌿 Optimal aerobic and fat oxidation zone. Ideal for building metabolic base.")
    else:
        recs.append("🚶 Good active recovery session. Helps blood flow and muscular recovery.")
        
    if cal_per_min > 12:
        recs.append(f"Impressive burn rate of {cal_per_min} kcal/min during {exercise_type}!")
    elif cal_per_min < 5 and duration > 30:
        recs.append(f"Steady pace maintained. To elevate caloric burn, introduce short 30-second interval surges.")
        
    return " ".join(recs)

def predict_calories(user_input, model_name=None, daily_goal=500.0):
    """
    End-to-end prediction method.
    user_input: dict with Age, Gender, Height, Weight, Duration, Heart_Rate, Body_Temp, Exercise_Type, Activity_Level
    """
    model, selected_model_name = load_model(model_name)
    
    if selected_model_name == "Deep Neural Network (DNN)":
        features_array = transform_single_input(user_input, MODELS_DIR, return_df=False)
        prediction_val = float(model.predict(features_array, verbose=0)[0][0])
    else:
        features_df = transform_single_input(user_input, MODELS_DIR, return_df=True)
        prediction_val = float(model.predict(features_df)[0])
        
    # Ensure realistic non-negative floor
    estimated_calories = max(round(prediction_val, 1), 5.0)
    
    # Calculate physiological metrics
    zone_info = calculate_intensity_and_zone(
        user_input['Heart_Rate'], 
        user_input['Age'], 
        user_input['Duration']
    )
    
    # Goal contribution
    goal_pct = round(min((estimated_calories / max(daily_goal, 1.0)) * 100.0, 100.0), 1)
    
    # Recommendation
    recommendation = generate_ai_recommendation(
        estimated_calories,
        user_input['Duration'],
        user_input['Heart_Rate'],
        user_input['Age'],
        user_input['Exercise_Type'],
        zone_info['intensity']
    )
    
    meta = get_model_metadata()
    model_metrics = meta.get("models", {}).get(selected_model_name, {})
    
    return {
        "calories": estimated_calories,
        "model_used": selected_model_name,
        "model_metrics": model_metrics,
        "intensity": zone_info['intensity'],
        "zone_name": zone_info['zone_name'],
        "pct_max_hr": zone_info['pct_max_hr'],
        "max_hr": zone_info['max_hr'],
        "intensity_color": zone_info['color'],
        "goal_contribution_pct": goal_pct,
        "recommendation": recommendation,
        "cal_per_min": round(estimated_calories / max(user_input['Duration'], 1), 1)
    }

if __name__ == "__main__":
    sample_input = {
        "Age": 28,
        "Gender": "male",
        "Height": 178.0,
        "Weight": 75.0,
        "Duration": 45,
        "Heart_Rate": 152,
        "Body_Temp": 38.8,
        "Exercise_Type": "Running",
        "Activity_Level": "Moderately Active"
    }
    res = predict_calories(sample_input)
    print("Prediction Result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
