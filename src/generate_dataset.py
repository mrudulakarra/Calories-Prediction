"""
Dataset Generation for Calorie Burnt Prediction
Generates a realistic physiological dataset with 15,000 records combining 
MET principles, Keytel heart rate formulas, thermogenesis, and activity classifications.
"""

import numpy as np
import pandas as pd
import os

def generate_calorie_dataset(num_samples=15000, random_seed=42):
    np.random.seed(random_seed)
    
    exercise_types = [
        "Walking", "Running", "Cycling", "Swimming", 
        "Gym", "HIIT", "Yoga", "Strength Training", "Other"
    ]
    
    activity_levels = [
        "Sedentary", "Lightly Active", "Moderately Active", 
        "Very Active", "Extremely Active"
    ]
    
    # MET baseline factors per exercise
    met_factors = {
        "Walking": 3.8,
        "Running": 9.8,
        "Cycling": 7.5,
        "Swimming": 8.0,
        "Gym": 6.0,
        "HIIT": 11.2,
        "Yoga": 3.0,
        "Strength Training": 5.5,
        "Other": 5.0
    }
    
    # Activity level multipliers
    activity_multipliers = {
        "Sedentary": 0.92,
        "Lightly Active": 0.98,
        "Moderately Active": 1.0,
        "Very Active": 1.05,
        "Extremely Active": 1.10
    }

    genders = np.random.choice(["male", "female"], size=num_samples, p=[0.51, 0.49])
    ages = np.random.randint(18, 72, size=num_samples)
    
    # Generate Heights & Weights correlated with gender
    heights = []
    weights = []
    for g in genders:
        if g == "male":
            h = np.random.normal(176, 8.5)
            w = np.random.normal(78, 12.0)
        else:
            h = np.random.normal(163, 7.5)
            w = np.random.normal(63, 10.5)
        heights.append(round(np.clip(h, 145, 208), 1))
        weights.append(round(np.clip(w, 42, 130), 1))
        
    heights = np.array(heights)
    weights = np.array(weights)
    
    durations = np.random.exponential(scale=22, size=num_samples) + 5
    durations = np.clip(np.round(durations), 1, 90).astype(int)
    
    selected_exercises = np.random.choice(exercise_types, size=num_samples, p=[
        0.16, 0.18, 0.14, 0.10, 0.12, 0.12, 0.08, 0.08, 0.02
    ])
    
    selected_activity = np.random.choice(activity_levels, size=num_samples, p=[
        0.15, 0.25, 0.35, 0.18, 0.07
    ])
    
    heart_rates = []
    body_temps = []
    calories = []
    
    for i in range(num_samples):
        g = genders[i]
        age = ages[i]
        w = weights[i]
        h = heights[i]
        dur = durations[i]
        ex = selected_exercises[i]
        act = selected_activity[i]
        
        met = met_factors[ex]
        act_mult = activity_multipliers[act]
        
        # Base resting & workout heart rate calculation
        max_hr = 220 - age
        target_intensity = np.clip((met / 12.0) + np.random.normal(0, 0.08), 0.35, 0.95)
        hr = int(np.clip(70 + (max_hr - 70) * target_intensity + (dur / 90.0) * 8.0, 65, max_hr + 5))
        heart_rates.append(hr)
        
        # Body Temperature increases with duration and intensity (36.5 to 41.0 C)
        temp_rise = (dur / 90.0) * 2.2 + (hr / 200.0) * 1.5 + np.random.normal(0, 0.2)
        body_temp = round(float(np.clip(37.0 + temp_rise, 36.8, 41.5)), 1)
        body_temps.append(body_temp)
        
        # Keytel Physiological Calorie Equation + Thermogenesis & Exercise adjustments
        if g == "male":
            cal_rate_per_min = ((-55.0969 + (0.6309 * hr) + (0.1988 * w) + (0.2017 * age)) / 4.184)
        else:
            cal_rate_per_min = ((-20.4022 + (0.4472 * hr) - (0.1263 * w) + (0.074 * age)) / 4.184)
            
        cal_rate_per_min = max(cal_rate_per_min, 2.0)
        
        # Apply MET factor and thermogenic bonus
        met_weight = 0.5 * (met / 6.0) + 0.5
        temp_factor = 1.0 + max(0.0, (body_temp - 37.0) * 0.035)
        
        total_cal = cal_rate_per_min * dur * met_weight * temp_factor * act_mult
        total_cal += np.random.normal(0, 4.0)
        total_cal = max(round(total_cal, 1), 5.0)
        calories.append(total_cal)
        
    df = pd.DataFrame({
        "User_ID": np.random.randint(10000000, 99999999, size=num_samples),
        "Gender": genders,
        "Age": ages,
        "Height": heights,
        "Weight": weights,
        "Duration": durations,
        "Heart_Rate": heart_rates,
        "Body_Temp": body_temps,
        "Exercise_Type": selected_exercises,
        "Activity_Level": selected_activity,
        "Calories": calories
    })
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/calories.csv", index=False)
    print(f"Dataset generated successfully with {len(df)} records at data/calories.csv")
    print(df.head(5))
    return df

if __name__ == "__main__":
    generate_calorie_dataset()
