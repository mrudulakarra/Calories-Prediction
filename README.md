# 🔥 Calories Burnt Prediction AI
### AI-Powered Personal Fitness, Calorie & Activity Intelligence

A state-of-the-art fitness and calorie intelligence application powered by **Python, Streamlit, Scikit-learn, TensorFlow/Keras, OpenCV, Plotly, SQLite, and ReportLab**.

---

## 🌟 Key Features

* **🔥 AI & Deep Learning Calorie Prediction Engine**:
  * Trains and benchmarks **Linear Regression**, **Random Forest**, **Gradient Boosting (GBDT)**, and a multi-layer **Deep Neural Network (DNN)** with TensorFlow/Keras.
  * Real-time physiological inference using Age, Gender, Height, Weight, Duration, Heart Rate, Body Temperature, Exercise Type, and Activity Level.
  * Heart rate zone classification, MET estimation, and dynamic AI recovery advice.

* **🏠 Executive Fitness Dashboard**:
  * Real-time metrics computed directly from SQLite: Calories burned today, active workout duration, sessions completed, daily/weekly goal progress %, and active streak tracking.
  * Interactive Plotly weekly burn trends and exercise distribution charts.

* **📷 Camera Access & Live Capture**:
  * Live camera snapshot capture with graceful fallback image upload.
  * Rapid posture biomechanics check, food calorie estimation, and progress photo storage.

* **🖼️ Computer Vision Image Analysis**:
  * **Food Nutrition**: Color space segmentation and visual feature heuristics estimating meal category, portion size, and macronutrients (Protein, Carbs, Fat, Fiber).
  * **Posture Biomechanics**: OpenCV Canny edge and contour detection with center plumb-line alignment, symmetry scoring, and kinetic chain feedback.
  * **Progress Photo Gallery**: Stored Before/Progress/Current photos with logged bodyweight.

* **🎥 Workout Video Motion Tracking**:
  * Upload MP4, MOV, AVI, or WEBM videos.
  * Frame sampling and optical flow motion tracking.
  * Repetition counting algorithm detecting peak-trough motion cycles (e.g. Squats, Push-ups, Jumping Jacks).
  * Cadence waveform visualization in Plotly.

* **📊 Complete Workout History & Audit**:
  * Filter by exercise type, input source (Manual, Camera, Image, Video), and search notes.
  * Export historical logs to **CSV** or **JSON**.

* **🎯 Dynamic Goals & Gamification Hub**:
  * Configurable daily & weekly calorie, duration, and frequency goals with live progress bars.
  * **12 Achievement Badges** unlocked automatically based on verified database workout history.

* **⏰ Alarms, Reminders & AI Smart Prompts**:
  * Scheduled reminders for workouts, hydration, meal times, and daily targets.
  * **Modular 3-Layer Notification System**:
    * **Layer 1**: In-App UI notifications.
    * **Layer 2**: Native Browser Web Notification API via JavaScript.
    * **Layer 3**: Email alerts (SMTP) & Push Notifications via Webhooks (Pushbullet/Pushover).
  * **AI Smart Reminders**: Contextual prompts detecting daily calorie deficits and habitual workout times.

* **📄 Professional PDF Fitness Reports (ReportLab)**:
  * Compiles complete athlete profile, executive summaries, model benchmark comparisons, recent workout logs, AI insights, and unlocked badges into a styled PDF with download support.

* **👤 Profile & Basal Metabolism (BMR/BMI)**:
  * Athlete profile management, BMI calculation, and Mifflin-St Jeor Basal Metabolic Rate (BMR) computation.

---

## 📁 Project Structure

```text
calories-burnt-ai/
│
├── app.py                     # Master Streamlit Application
│
├── data/
│   └── calories.csv           # Physiological Calorie Dataset
│
├── models/
│   ├── random_forest.pkl      # Trained Random Forest Regressor
│   ├── gradient_boosting.pkl  # Trained Gradient Boosting Regressor
│   ├── linear_regression.pkl  # Trained Linear Regression
│   ├── scaler.pkl             # Feature StandardScaler
│   ├── feature_info.pkl       # Feature & Category Metadata
│   ├── calories_dnn.keras     # Deep Learning Keras Neural Network
│   └── model_metadata.json    # Benchmark Evaluation Metrics (MAE, RMSE, R²)
│
├── src/
│   ├── generate_dataset.py    # Synthetic physiological data generator
│   ├── preprocessing.py       # Data pipeline, encoding & scaling
│   ├── train_ml.py            # ML Model training (LR, RF, GBDT)
│   ├── train_dl.py            # Deep Learning DNN training (Keras)
│   ├── prediction.py          # Real-time inference engine & intensity zones
│   ├── database.py            # SQLite schema, tables & queries
│   ├── camera.py              # Camera and image capture management
│   ├── image_analysis.py      # Computer Vision food & posture analysis
│   ├── video_analysis.py      # Video optical flow & rep counting
│   ├── notifications.py       # 3-Layer modular notification service
│   ├── alarms.py              # Alarms & AI smart reminder logic
│   ├── achievements.py        # Gamification & badge unlocking engine
│   ├── insights.py            # Statistical habit intelligence
│   └── reports.py             # ReportLab PDF report compiler
│
├── database/
│   └── fitness.db             # SQLite Database
│
├── reports/                   # Compiled PDF Reports
├── uploads/                   # Uploaded & captured photos/videos
├── assets/
│   └── custom.css             # High-end Dark Neon UI styling
│
├── .env                       # Environment credentials (API / SMTP)
├── .gitignore                 # Security rules (secrets ignored)
├── requirements.txt           # Python dependencies
└── README.md                  # Project Documentation
```

---

## 🚀 Running the Project (Windows PowerShell)

### 1. Setup Virtual Environment & Install Dependencies
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```powershell
python src/generate_dataset.py
python src/train_ml.py
python src/train_dl.py
```

### 3. Launch the Streamlit Application
```powershell
streamlit run app.py
```

---

## 🔒 Security Best Practices

* Never hardcode API keys, SMTP passwords, or webhooks directly in Python code.
* Store all credentials in `.env` or Streamlit Secrets.
* `.gitignore` prevents sensitive `.env`, keys, credentials, and uploaded files from being tracked in source control.

---

## 🛡️ Medical & Exercise Disclaimer

All calorie predictions, computer vision posture feedbacks, and food nutritional estimates are intended for personal fitness tracking and informational purposes only. Consult qualified healthcare professionals or certified nutritionists for medical advice or clinical dietary planning.
