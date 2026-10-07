"""
================================================================================
CALORIES BURNT PREDICTION AI — ADVANCED AI FITNESS ASSISTANT
AI-Powered Personal Fitness, Calorie & Activity Intelligence
================================================================================
"""

import os
import sys
import json
import base64
from datetime import datetime, date, timedelta
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure root directory is accessible for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.database import (
    init_db, get_user_profile, update_user_profile, get_dashboard_metrics,
    get_all_workouts, log_workout, delete_workout, get_user_goals,
    save_user_goal, delete_user_goal, get_user_reminders, add_reminder,
    toggle_reminder, delete_reminder, save_progress_photo, get_progress_photos,
    get_user_settings, update_user_settings
)
from src.prediction import predict_calories, get_model_metadata
from src.achievements import get_user_achievements_status, check_and_unlock_achievements
from src.insights import compute_user_insights
from src.alarms import get_upcoming_alarms, generate_smart_alarm_suggestions
from src.notifications import NotificationService
from src.reports import generate_pdf_report
from src.camera import save_uploaded_or_captured_image
from src.image_analysis import analyze_food_image, analyze_exercise_posture
from src.video_analysis import analyze_workout_video

# -----------------------------------------------------------------------------
# App Configuration & Theme Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Calories Burnt AI — Fitness Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Load custom CSS
css_file = os.path.join("assets", "custom.css")
if os.path.exists(css_file):
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "new_achievement_alert" not in st.session_state:
    st.session_state.new_achievement_alert = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

user_id = st.session_state.user_id
user = get_user_profile(user_id)
settings = get_user_settings(user_id)

# Plotly theme layout standard
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor='rgba(17, 24, 39, 0)',
    plot_bgcolor='rgba(17, 24, 39, 0.4)',
    font=dict(color='#94A3B8', family='Plus Jakarta Sans, sans-serif'),
    margin=dict(l=20, r=20, t=35, b=20),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', showgrid=True),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.06)', showgrid=True),
    legend=dict(font=dict(color='#F8FAFC', size=11))
)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
        <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FFD166 100%); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);">
            🔥
        </div>
        <div>
            <h2 style="font-size: 1.25rem; margin: 0; color: #FFFFFF; font-weight: 800; letter-spacing: -0.01em;">CALORIE AI</h2>
            <span style="font-size: 0.72rem; color: #00F2FE; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;">Fitness Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # User Profile Pill
    if user:
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 10px 14px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 0.9rem;">{user['name']}</div>
                <div style="color: #94A3B8; font-size: 0.75rem;">Goal: {user['daily_goal']} kcal/day</div>
            </div>
            <span class="badge-pill badge-emerald">Active</span>
        </div>
        """, unsafe_allow_html=True)
        
    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔥 Calorie Prediction",
            "📷 Camera",
            "🖼️ Image Analysis",
            "🎥 Video Analysis",
            "📊 History",
            "🎯 Goals",
            "⏰ Alarms & Reminders",
            "🏆 Achievements",
            "📄 Reports",
            "🤖 AI Insights",
            "👤 Profile",
            "⚙️ Settings"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("🚀 Version 2.0 • ML + Deep Learning")
    st.caption("⚡ Models: DNN, GBDT, RF, Linear")

# Check for new achievement unlocks on every view
newly_unlocked = check_and_unlock_achievements(user_id)
if newly_unlocked:
    for ach in newly_unlocked:
        st.toast(f"🎉 Achievement Unlocked: {ach['icon']} {ach['title']}!", icon="🏆")

# =============================================================================
# 1. 🏠 DASHBOARD
# =============================================================================
if nav_selection == "🏠 Dashboard":
    metrics = get_dashboard_metrics(user_id)
    workouts_df = get_all_workouts(user_id)
    smart_alarms = generate_smart_alarm_suggestions(user_id)
    
    # Top Welcome Header
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title(f"Welcome back, {user['name'] if user else 'Athlete'}! 👋")
        st.markdown("<p style='color: #94A3B8; font-size: 1rem; margin-top: -10px;'>Here is your real-time physiological fitness overview and calorie intelligence.</p>", unsafe_allow_html=True)
    with col_head2:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <span class="badge-pill badge-flame" style="font-size: 0.85rem; padding: 6px 14px;">
                🔥 {metrics['current_streak']} Day Streak
            </span>
        </div>
        """, unsafe_allow_html=True)
        
    # Smart Reminder Alert (if any active recommendation)
    if smart_alarms:
        top_alarm = smart_alarms[0]
        st.markdown(f"""
        <div class="smart-alarm-box">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h4 style="margin: 0; color: #00F2FE; font-size: 1rem;">{top_alarm['title']}</h4>
                    <p style="margin: 4px 0 0 0; color: #E2E8F0; font-size: 0.88rem;">{top_alarm['message']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 6 Premium Metric Cards Row
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    
    with mcol1:
        st.markdown(f"""
        <div class="fitness-metric-card metric-card-flame">
            <div class="metric-label">Today's Calories</div>
            <div class="metric-value">{metrics['today_calories']} <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">kcal</span></div>
            <div class="metric-delta" style="color: #00F5A0;">🔥 Burn Output</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol2:
        st.markdown(f"""
        <div class="fitness-metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{metrics['today_duration']} <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">min</span></div>
            <div class="metric-delta" style="color: #00F2FE;">⏱ Active Time</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol3:
        st.markdown(f"""
        <div class="fitness-metric-card metric-card-purple">
            <div class="metric-label">Workouts</div>
            <div class="metric-value">{metrics['today_workouts']}</div>
            <div class="metric-delta" style="color: #A78BFA;">🏃 Sessions Today</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol4:
        st.markdown(f"""
        <div class="fitness-metric-card metric-card-amber">
            <div class="metric-label">Daily Goal</div>
            <div class="metric-value">{metrics['daily_goal_pct']}%</div>
            <div class="metric-delta" style="color: #FFD166;">🎯 {metrics['today_calories']} / {metrics['daily_goal']} kcal</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol5:
        st.markdown(f"""
        <div class="fitness-metric-card">
            <div class="metric-label">Active Streak</div>
            <div class="metric-value">{metrics['current_streak']} <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">days</span></div>
            <div class="metric-delta" style="color: #00F2FE;">⚡ Consistency</div>
        </div>
        """, unsafe_allow_html=True)
        
    with mcol6:
        st.markdown(f"""
        <div class="fitness-metric-card metric-card-flame">
            <div class="metric-label">Weekly Burn</div>
            <div class="metric-value">{metrics['weekly_calories']:,.0f} <span style="font-size: 0.9rem; font-weight: 500; color: #94A3B8;">kcal</span></div>
            <div class="metric-delta" style="color: #00F5A0;">📈 {metrics['weekly_goal_pct']}% of week goal</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    
    # Visual Analytics Charts
    chart_col1, chart_col2 = st.columns([3, 2])
    
    with chart_col1:
        st.subheader("Weekly Calorie Burn & Goal Tracking")
        if not workouts_df.empty:
            # Aggregate last 7 days calories
            now_dt = datetime.now().date()
            seven_days = [(now_dt - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
            weekly_agg = workouts_df[workouts_df['date'].isin(seven_days)].groupby('date')['calories'].sum().reindex(seven_days, fill_value=0.0).reset_index()
            weekly_agg['Day'] = pd.to_datetime(weekly_agg['date']).dt.strftime('%a (%d %b)')
            
            fig_week = go.Figure()
            # Bar chart for calories
            fig_week.add_trace(go.Bar(
                x=weekly_agg['Day'],
                y=weekly_agg['calories'],
                name='Calories Burned',
                marker=dict(
                    color=weekly_agg['calories'],
                    colorscale=[[0, '#00F2FE'], [1, '#FF6B6B']],
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                ),
                hovertemplate="<b>%{x}</b><br>Burned: %{y:.1f} kcal<extra></extra>"
            ))
            # Daily Target Line
            fig_week.add_trace(go.Scatter(
                x=weekly_agg['Day'],
                y=[metrics['daily_goal']] * len(weekly_agg),
                name='Daily Goal',
                mode='lines',
                line=dict(color='#FFD166', width=2, dash='dash'),
                hovertemplate="Daily Target: %{y} kcal<extra></extra>"
            ))
            fig_week.update_layout(**PLOTLY_DARK_LAYOUT, height=330)
            st.plotly_chart(fig_week, use_container_width=True)
        else:
            st.info("No workout history recorded yet.")
            
    with chart_col2:
        st.subheader("Exercise Distribution")
        if not workouts_df.empty:
            ex_dist = workouts_df.groupby('exercise')['calories'].sum().reset_index()
            fig_pie = px.pie(
                ex_dist,
                names='exercise',
                values='calories',
                hole=0.55,
                color_discrete_sequence=['#00F2FE', '#00F5A0', '#FF6B6B', '#FFD166', '#8B5CF6', '#EC4899', '#38BDF8']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#111827', width=2)))
            fig_pie.update_layout(**PLOTLY_DARK_LAYOUT, height=330, showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No exercise distribution data available.")
            
    # Recent Workouts & Upcoming Reminders Row
    bottom_col1, bottom_col2 = st.columns([3, 2])
    
    with bottom_col1:
        st.subheader("Recent Workout Logs")
        if not workouts_df.empty:
            recent_display = workouts_df[['date', 'exercise', 'duration', 'heart_rate', 'calories', 'intensity', 'input_source']].head(5).copy()
            recent_display.columns = ['Date', 'Exercise', 'Duration (min)', 'Heart Rate (bpm)', 'Calories (kcal)', 'Intensity', 'Source']
            st.dataframe(recent_display, use_container_width=True, hide_index=True)
        else:
            st.info("No workouts logged yet.")
            
    with bottom_col2:
        st.subheader("Upcoming Alarms & Reminders")
        reminders_df = get_upcoming_alarms(user_id)
        if not reminders_df.empty:
            for _, rem in reminders_df.head(3).iterrows():
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem;">⏰ {rem['title']}</div>
                        <div style="color: #94A3B8; font-size: 0.78rem;">{rem['repeat_type']} • {rem['time']} ({rem['notification_type']})</div>
                    </div>
                    <span class="badge-pill badge-cyan">Scheduled</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active reminders scheduled. Set one in Alarms & Reminders.")

# =============================================================================
# 2. 🔥 CALORIE PREDICTION
# =============================================================================
elif nav_selection == "🔥 Calorie Prediction":
    st.title("🔥 AI Calorie Prediction Engine")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Estimate calories burned using trained physiological Machine Learning & Deep Learning Neural Networks.</p>", unsafe_allow_html=True)
    
    meta = get_model_metadata()
    best_m = meta.get("best_model", "Deep Neural Network (DNN)")
    
    model_options = ["Auto (Best Model)", "Deep Neural Network (DNN)", "Gradient Boosting", "Random Forest", "Linear Regression"]
    
    # Input Form
    with st.container():
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            selected_model_option = st.selectbox(
                "Select Prediction Model",
                model_options,
                index=0,
                help=f"Auto mode uses the highest R² model: {best_m}"
            )
        with col_m2:
            st.markdown(f"""
            <div style="padding-top: 28px;">
                <span class="badge-pill badge-cyan">⭐ Best Model: {best_m}</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # User Physiological Inputs
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            age_val = st.number_input("Age (years)", min_value=15, max_value=85, value=int(user['age']) if user else 28)
            gender_val = st.selectbox("Gender", ["male", "female"], index=0 if user and user['gender'] == 'male' else 1)
            height_val = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=float(user['height']) if user else 178.0, step=0.5)
            
        with pcol2:
            weight_val = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=float(user['weight']) if user else 75.0, step=0.5)
            duration_val = st.slider("Workout Duration (minutes)", min_value=1, max_value=120, value=45)
            heart_rate_val = st.slider("Average Heart Rate (bpm)", min_value=60, max_value=210, value=150)
            
        with pcol3:
            body_temp_val = st.slider("Body Temperature (°C)", min_value=36.0, max_value=41.5, value=38.8, step=0.1)
            exercise_val = st.selectbox("Exercise Type", ["Walking", "Running", "Cycling", "Swimming", "Gym", "HIIT", "Yoga", "Strength Training", "Other"], index=1)
            activity_val = st.selectbox("Baseline Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"], index=2)
            
        predict_btn = st.button("🔥 Calculate Estimated Calorie Burn", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Process Prediction
    if predict_btn or st.session_state.last_prediction is not None:
        if predict_btn:
            input_dict = {
                "Age": age_val,
                "Gender": gender_val,
                "Height": height_val,
                "Weight": weight_val,
                "Duration": duration_val,
                "Heart_Rate": heart_rate_val,
                "Body_Temp": body_temp_val,
                "Exercise_Type": exercise_val,
                "Activity_Level": activity_val
            }
            pred_result = predict_calories(
                input_dict,
                model_name=selected_model_option,
                daily_goal=user['daily_goal'] if user else 500.0
            )
            st.session_state.last_prediction = {
                "input": input_dict,
                "result": pred_result
            }
            
        pred_data = st.session_state.last_prediction
        p_res = pred_data["result"]
        p_inp = pred_data["input"]
        
        # Hero Prediction Display Card
        st.markdown(f"""
        <div class="prediction-hero-card">
            <div style="font-size: 0.9rem; font-weight: 700; color: #00F2FE; text-transform: uppercase; letter-spacing: 0.08em;">
                ⚡ Model Inference: {p_res['model_used']} (R²: {p_res['model_metrics'].get('r2', 'N/A')})
            </div>
            <div class="prediction-calorie-number">{p_res['calories']} <span style="font-size: 2rem; color: #FFD166;">kcal</span></div>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                <span class="badge-pill badge-flame">Intensity: {p_res['intensity']}</span>
                <span class="badge-pill badge-cyan">{p_res['zone_name']}</span>
                <span class="badge-pill badge-emerald">Burn Rate: {p_res['cal_per_min']} kcal/min</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Insights & Goal Contribution
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            st.subheader("🎯 Goal Contribution & Intensity")
            st.write(f"This session contributes **{p_res['goal_contribution_pct']}%** toward your daily target of {user['daily_goal'] if user else 500} kcal.")
            st.progress(min(p_res['goal_contribution_pct'] / 100.0, 1.0))
            
            st.write(f"**Max Heart Rate Percentage:** {p_res['pct_max_hr']}% (Max HR: {p_res['max_hr']} bpm)")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with gcol2:
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            st.subheader("🤖 AI Recommendation")
            st.info(p_res['recommendation'])
            
            # Historical Comparison
            workouts_df = get_all_workouts(user_id)
            if not workouts_df.empty:
                same_ex = workouts_df[workouts_df['exercise'] == p_inp['Exercise_Type']]
                if not same_ex.empty:
                    avg_prev = same_ex['calories'].mean()
                    diff_pct = round(((p_res['calories'] - avg_prev) / avg_prev) * 100, 1)
                    if diff_pct >= 0:
                        st.success(f"📈 This session burns **{diff_pct}% more** calories than your previous average for {p_inp['Exercise_Type']} ({round(avg_prev, 1)} kcal).")
                    else:
                        st.info(f"📊 Your burn is **{abs(diff_pct)}% lower** than your historical average for {p_inp['Exercise_Type']} ({round(avg_prev, 1)} kcal).")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Log to Database Action
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            workout_notes = st.text_input("Workout Notes (optional)", placeholder="e.g. Fast interval run at the local track")
        with save_col2:
            st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Save to Workout History", type="primary", use_container_width=True):
                w_id = log_workout(
                    user_id=user_id,
                    exercise=p_inp['Exercise_Type'],
                    duration=p_inp['Duration'],
                    heart_rate=p_inp['Heart_Rate'],
                    body_temp=p_inp['Body_Temp'],
                    calories=p_res['calories'],
                    intensity=p_res['intensity'],
                    input_source="Manual",
                    notes=workout_notes
                )
                st.success(f"✅ Workout successfully logged to SQLite database (Workout ID #{w_id})!")
                # Evaluate achievements
                new_achs = check_and_unlock_achievements(user_id)
                if new_achs:
                    for a in new_achs:
                        st.balloons()
                        st.toast(f"🏆 NEW BADGE UNLOCKED: {a['title']}!", icon="🎉")

# =============================================================================
# 3. 📷 CAMERA
# =============================================================================
elif nav_selection == "📷 Camera":
    st.title("📷 Camera Snapshot & Fitness Capture")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Capture live photos using your webcam/device camera for fitness posture analysis, workout documentation, or food tracking.</p>", unsafe_allow_html=True)
    
    cam_mode = st.radio(
        "Capture Category",
        ["Exercise / Posture Snapshot", "Food & Nutrition Capture", "Progress Photo", "Profile Picture"],
        horizontal=True
    )
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    cam_picture = st.camera_input("📷 Take Photo")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Fallback upload option
    if not cam_picture:
        st.caption("💡 If camera access is unavailable or denied, you can use the fallback uploader below:")
        cam_picture = st.file_uploader("Upload Image Fallback", type=["jpg", "jpeg", "png", "webp"], key="cam_fallback")
        
    if cam_picture:
        st.success("✅ Image captured successfully!")
        preview_col1, preview_col2 = st.columns([1, 1])
        
        with preview_col1:
            st.subheader("Captured Preview")
            st.image(cam_picture, use_container_width=True)
            
        with preview_col2:
            st.subheader("AI Analysis & Actions")
            
            if cam_mode == "Exercise / Posture Snapshot":
                if st.button("🧘 Analyze Posture & Alignment", type="primary", use_container_width=True):
                    with st.spinner("Analyzing body alignment with OpenCV..."):
                        posture_res = analyze_exercise_posture(cam_picture)
                        st.session_state.cam_posture_res = posture_res
                        
                if "cam_posture_res" in st.session_state:
                    p_res = st.session_state.cam_posture_res
                    st.image(p_res['annotated_image'], caption="Biomechanics Overlay", use_container_width=True)
                    st.metric("Posture Alignment Score", f"{p_res['symmetry_score']}%")
                    st.write(f"**Detected Pose:** {p_res['exercise_type_guess']}")
                    for tip in p_res['feedback']:
                        st.write(tip)
                            
            elif cam_mode == "Food & Nutrition Capture":
                if st.button("🥗 Analyze Food & Estimate Calories", type="primary", use_container_width=True):
                    with st.spinner("Analyzing nutritional content with Computer Vision..."):
                        food_res = analyze_food_image(cam_picture)
                        st.session_state.cam_food_res = food_res
                        
                if "cam_food_res" in st.session_state:
                    f_res = st.session_state.cam_food_res
                    st.markdown(f"""
                    <div class="fitness-metric-card metric-card-amber" style="margin: 14px 0;">
                        <div class="metric-label">Estimated Caloric Intake</div>
                        <div class="metric-value">{f_res['estimated_calories']} <span style="font-size: 1.1rem; color: #94A3B8;">kcal</span></div>
                        <div class="metric-delta" style="color: #00F5A0;">🍽️ {f_res['portion_size']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**Identified Item:** `{f_res['category']}`")
                    st.write(f"**Nutritional Rating:** {f_res['health_rating']}")
                    
                    m = f_res['macros']
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Protein", f"{m['protein_g']}g")
                    mc2.metric("Carbs", f"{m['carbs_g']}g")
                    mc3.metric("Fat", f"{m['fat_g']}g")
                    mc4.metric("Fiber", f"{m['fiber_g']}g")
                    
                    st.caption(f_res['disclaimer'])
                        
            elif cam_mode == "Progress Photo":
                p_type = st.selectbox("Progress Type", ["Current", "Progress", "Before"])
                p_weight = st.number_input("Current Bodyweight (kg)", value=float(user['weight']) if user else 75.0)
                p_notes = st.text_input("Notes", placeholder="e.g. Week 4 check-in")
                if st.button("💾 Save to Progress Gallery", type="primary", use_container_width=True):
                    saved_path, _ = save_uploaded_or_captured_image(cam_picture, "images")
                    save_progress_photo(user_id, p_type, saved_path, datetime.now().strftime("%Y-%m-%d"), p_weight, p_notes)
                    st.success("✅ Progress photo saved to gallery!")
                    
            elif cam_mode == "Profile Picture":
                if st.button("👤 Set as Profile Photo", type="primary", use_container_width=True):
                    saved_path, _ = save_uploaded_or_captured_image(cam_picture, "profile")
                    update_user_profile(
                        user_id, user['name'], user['age'], user['gender'],
                        user['height'], user['weight'], user['fitness_goal'],
                        user['activity_level'], user['daily_goal'], user['weekly_goal'],
                        profile_pic=saved_path
                    )
                    st.success("✅ Profile picture updated successfully!")

# =============================================================================
# 4. 🖼️ IMAGE ANALYSIS
# =============================================================================
elif nav_selection == "🖼️ Image Analysis":
    st.title("🖼️ Computer Vision & Image Intelligence")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Analyze meals, posture biomechanics, or track your physical transformation over time.</p>", unsafe_allow_html=True)
    
    img_tab1, img_tab2, img_tab3 = st.tabs(["🥗 Food & Calorie Estimation", "🧘 Exercise & Posture Biomechanics", "📸 Progress Photo Gallery"])
    
    # Tab 1: Food Analysis
    with img_tab1:
        st.subheader("Food Nutrition & Calorie Estimator")
        uploaded_food = st.file_uploader("Upload meal photo (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key="food_uploader")
        
        if uploaded_food:
            fcol1, fcol2 = st.columns([1, 1])
            with fcol1:
                st.image(uploaded_food, caption="Uploaded Meal", use_container_width=True)
            with fcol2:
                with st.spinner("Processing visual features with OpenCV..."):
                    food_data = analyze_food_image(uploaded_food)
                    
                st.markdown(f"""
                <div class="fitness-metric-card metric-card-amber" style="margin-bottom: 16px;">
                    <div class="metric-label">Estimated Calorie Content</div>
                    <div class="metric-value">{food_data['estimated_calories']} <span style="font-size: 1.1rem; color: #94A3B8;">kcal</span></div>
                    <div class="metric-delta" style="color: #00F5A0;">🍽️ {food_data['portion_size']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(f"**Identified Dish Category:** {food_data['category']}")
                st.write(f"**Nutritional Rating:** {food_data['health_rating']}")
                
                macros = food_data['macros']
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Protein", f"{macros['protein_g']}g")
                mcol2.metric("Carbs", f"{macros['carbs_g']}g")
                mcol3.metric("Fat", f"{macros['fat_g']}g")
                mcol4.metric("Fiber", f"{macros['fiber_g']}g")
                
                st.info(food_data['disclaimer'])
                
    # Tab 2: Posture Biomechanics
    with img_tab2:
        st.subheader("Exercise Form & Posture Alignment")
        uploaded_exercise = st.file_uploader("Upload exercise snapshot (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key="exercise_uploader")
        
        if uploaded_exercise:
            ecol1, ecol2 = st.columns([1, 1])
            with ecol1:
                st.image(uploaded_exercise, caption="Original Pose", use_container_width=True)
            with ecol2:
                with st.spinner("Extracting body contours and alignment plumb-line..."):
                    post_data = analyze_exercise_posture(uploaded_exercise)
                    
                st.image(post_data['annotated_image'], caption="Biomechanics Analysis Overlay", use_container_width=True)
                st.metric("Posture Alignment Score", f"{post_data['symmetry_score']}%")
                st.write(f"**Observed Stance:** {post_data['exercise_type_guess']}")
                for tip in post_data['feedback']:
                    st.write(tip)
                    
    # Tab 3: Progress Photos Gallery
    with img_tab3:
        st.subheader("Physical Transformation & Progress Gallery")
        
        with st.expander("➕ Upload New Progress Photo", expanded=False):
            p_up = st.file_uploader("Upload progress photo", type=["png", "jpg", "jpeg", "webp"], key="prog_uploader")
            p_typ = st.selectbox("Stage", ["Before", "Progress", "Current"], key="p_stage")
            p_wt = st.number_input("Logged Weight (kg)", value=float(user['weight']) if user else 75.0, key="p_weight_in")
            p_nt = st.text_input("Photo Notes", placeholder="e.g. Month 2 physique update", key="p_notes_in")
            
            if st.button("Save Photo to Gallery", type="primary") and p_up:
                saved_p, _ = save_uploaded_or_captured_image(p_up, "images")
                save_progress_photo(user_id, p_typ, saved_p, datetime.now().strftime("%Y-%m-%d"), p_wt, p_nt)
                st.success("✅ Photo added to progress records!")
                st.rerun()
                
        photos_df = get_progress_photos(user_id)
        if not photos_df.empty:
            st.write(f"Total Stored Photos: **{len(photos_df)}**")
            pcols = st.columns(3)
            for idx, (_, p_row) in enumerate(photos_df.iterrows()):
                with pcols[idx % 3]:
                    if os.path.exists(p_row['file_path']):
                        st.image(p_row['file_path'], use_container_width=True)
                        st.caption(f"📅 **{p_row['date']}** • Stage: `{p_row['photo_type']}` • Weight: {p_row['weight']} kg")
                        if p_row['notes']:
                            st.caption(f"📝 {p_row['notes']}")
        else:
            st.info("No progress photos stored yet. Upload your first photo above or via the Camera page!")

# =============================================================================
# 5. 🎥 VIDEO ANALYSIS
# =============================================================================
elif nav_selection == "🎥 Video Analysis":
    st.title("🎥 Workout Video Motion Tracking")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Analyze workout video clips using OpenCV optical flow, repetition cycle counting, and motion intensity estimation.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload Workout Video (MP4, MOV, AVI, WEBM)", type=["mp4", "mov", "avi", "webm"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_video:
        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            st.subheader("Video Preview")
            st.video(uploaded_video)
            
        with vcol2:
            st.subheader("Video Analytics Processing")
            analyze_vid_btn = st.button("⚡ Run Computer Vision Video Analysis", type="primary", use_container_width=True)
            
            if analyze_vid_btn:
                with st.spinner("Extracting frames and tracking optical flow motion waveforms..."):
                    video_bytes = uploaded_video.read()
                    v_res = analyze_workout_video(
                        video_bytes,
                        user_weight=user['weight'] if user else 75.0,
                        user_age=user['age'] if user else 28,
                        user_gender=user['gender'] if user else 'male'
                    )
                    st.session_state.video_result = v_res
                    
        if "video_result" in st.session_state:
            v_res = st.session_state.video_result
            
            st.markdown("---")
            st.subheader("📊 Workout Video Analysis Summary")
            
            vmcol1, vmcol2, vmcol3, vmcol4 = st.columns(4)
            vmcol1.metric("Detected Activity", v_res['detected_activity'])
            vmcol2.metric("Repetitions Counted", f"{v_res['repetitions']} reps")
            vmcol3.metric("Video Duration", f"{v_res['duration_sec']}s ({v_res['duration_min']} min)")
            vmcol4.metric("Estimated Calories", f"{v_res['estimated_calories']} kcal")
            
            # Interactive Motion Energy Waveform
            st.subheader("Motion Energy & Cadence Waveform")
            motion_curve = v_res['motion_curve']
            fig_motion = go.Figure()
            fig_motion.add_trace(go.Scatter(
                x=motion_curve['time_sec'],
                y=motion_curve['motion_energy'],
                mode='lines',
                name='Motion Energy (Cadence)',
                line=dict(color='#00F2FE', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(0, 242, 254, 0.15)'
            ))
            fig_motion.update_layout(
                **PLOTLY_DARK_LAYOUT,
                title="Cadence & Repetition Peaks Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Movement Intensity Energy",
                height=300
            )
            st.plotly_chart(fig_motion, use_container_width=True)
            
            st.info(v_res['summary'])
            
            # Log video workout to history
            if st.button("💾 Log Video Workout to History Database", type="primary"):
                w_id = log_workout(
                    user_id=user_id,
                    exercise=v_res['detected_activity'][:30],
                    duration=max(int(round(v_res['duration_min'])), 1),
                    heart_rate=145,
                    body_temp=38.4,
                    calories=v_res['estimated_calories'],
                    intensity=v_res['intensity'],
                    input_source="Video",
                    notes=f"CV Reps: {v_res['repetitions']} • Frames: {v_res['analyzed_frames']}"
                )
                st.success(f"✅ Video workout logged to history (ID #{w_id})!")
                check_and_unlock_achievements(user_id)

# =============================================================================
# 6. 📊 HISTORY
# =============================================================================
elif nav_selection == "📊 History":
    st.title("📊 Workout History & Exercise Logs")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Comprehensive audit trail of all manual, camera, image, and video workout sessions.</p>", unsafe_allow_html=True)
    
    workouts_df = get_all_workouts(user_id)
    
    if not workouts_df.empty:
        # Filter and Search Bar
        fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
        with fcol1:
            all_exercises = ["All"] + sorted(list(workouts_df['exercise'].unique()))
            filter_ex = st.selectbox("Filter by Exercise", all_exercises)
        with fcol2:
            all_sources = ["All"] + sorted(list(workouts_df['input_source'].unique()))
            filter_src = st.selectbox("Filter by Source", all_sources)
        with fcol3:
            search_query = st.text_input("Search Notes / Keyword", placeholder="Search...")
            
        filtered_df = workouts_df.copy()
        if filter_ex != "All":
            filtered_df = filtered_df[filtered_df['exercise'] == filter_ex]
        if filter_src != "All":
            filtered_df = filtered_df[filtered_df['input_source'] == filter_src]
        if search_query:
            filtered_df = filtered_df[
                filtered_df['exercise'].str.contains(search_query, case=False, na=False) |
                filtered_df['notes'].str.contains(search_query, case=False, na=False)
            ]
            
        # Summary KPI row for filtered data
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Sessions Found", len(filtered_df))
        k2.metric("Total Calories", f"{filtered_df['calories'].sum():,.1f} kcal")
        k3.metric("Total Duration", f"{filtered_df['duration'].sum()} min")
        k4.metric("Average HR", f"{filtered_df['heart_rate'].mean():.0f} bpm" if not filtered_df.empty else "0 bpm")
        
        # Interactive Time Series Plot
        if not filtered_df.empty:
            filtered_df['dt'] = pd.to_datetime(filtered_df['date'])
            fig_hist = px.scatter(
                filtered_df,
                x='date',
                y='calories',
                size='duration',
                color='exercise',
                hover_data=['time', 'heart_rate', 'intensity', 'input_source'],
                title="Historical Workout Timeline (Size = Duration, Color = Exercise)"
            )
            fig_hist.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        # Data table
        st.subheader("Workout Records")
        display_table = filtered_df[['id', 'date', 'time', 'exercise', 'duration', 'heart_rate', 'body_temp', 'calories', 'intensity', 'input_source', 'notes']]
        st.dataframe(display_table, use_container_width=True, hide_index=True)
        
        # Export Actions
        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])
        with exp_col1:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export CSV", data=csv_data, file_name=f"workout_history_{user_id}.csv", mime="text/csv")
        with exp_col2:
            json_data = filtered_df.to_json(orient="records", indent=2).encode('utf-8')
            st.download_button("📥 Export JSON", data=json_data, file_name=f"workout_history_{user_id}.json", mime="application/json")
        with exp_col3:
            del_id = st.number_input("Delete Workout ID", min_value=1, step=1, key="del_w_id")
            if st.button("🗑️ Delete Record", type="secondary"):
                delete_workout(del_id, user_id)
                st.success(f"Deleted workout ID #{del_id}")
                st.rerun()
    else:
        st.info("No workout records available in database yet. Log a workout in Calorie Prediction or Video Analysis!")

# =============================================================================
# 7. 🎯 GOALS
# =============================================================================
elif nav_selection == "🎯 Goals":
    st.title("🎯 Fitness Goals & Milestone Tracking")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Define daily and weekly targets for calories, duration, and exercise frequency.</p>", unsafe_allow_html=True)
    
    metrics = get_dashboard_metrics(user_id)
    goals_df = get_user_goals(user_id)
    
    # Active Goals Cards
    st.subheader("Current Active Goals")
    gcol1, gcol2 = st.columns(2)
    
    with gcol1:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.write("#### 🎯 Daily Calorie Target")
        st.write(f"Progress: **{metrics['today_calories']} / {metrics['daily_goal']} kcal** ({metrics['daily_goal_pct']}%)")
        st.progress(min(metrics['daily_goal_pct'] / 100.0, 1.0))
        rem_daily = max(metrics['daily_goal'] - metrics['today_calories'], 0)
        st.caption(f"{int(rem_daily)} kcal remaining to reach today's milestone." if rem_daily > 0 else "🎉 Daily goal achieved today!")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with gcol2:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.write("#### 🏆 Weekly Calorie Target")
        st.write(f"Progress: **{metrics['weekly_calories']:,.0f} / {metrics['weekly_goal']:,.0f} kcal** ({metrics['weekly_goal_pct']}%)")
        st.progress(min(metrics['weekly_goal_pct'] / 100.0, 1.0))
        rem_week = max(metrics['weekly_goal'] - metrics['weekly_calories'], 0)
        st.caption(f"{int(rem_week)} kcal remaining for the 7-day period." if rem_week > 0 else "🌟 Weekly goal crushed!")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Add New Goal Form
    with st.expander("➕ Set New Custom Goal", expanded=False):
        ncol1, ncol2, ncol3 = st.columns(3)
        with ncol1:
            g_type = st.selectbox("Goal Type", ["Daily Calorie Goal", "Weekly Calorie Goal", "Daily Duration Goal", "Weekly Frequency Goal"])
        with ncol2:
            g_target = st.number_input("Target Value (kcal / mins / sessions)", min_value=1.0, value=600.0, step=10.0)
        with ncol3:
            g_period = st.selectbox("Period", ["daily", "weekly", "monthly"])
            
        if st.button("Save New Goal", type="primary"):
            save_user_goal(user_id, g_type, g_target, g_period)
            st.success("✅ New goal registered!")
            st.rerun()
            
    # Existing Goals Table
    if not goals_df.empty:
        st.subheader("Configured Goals in Database")
        st.dataframe(goals_df[['id', 'goal_type', 'target_value', 'period', 'start_date', 'is_active']], use_container_width=True, hide_index=True)

# =============================================================================
# 8. ⏰ ALARMS & REMINDERS
# =============================================================================
elif nav_selection == "⏰ Alarms & Reminders":
    st.title("⏰ Alarms, Reminders & Smart AI Prompts")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Configure scheduled reminders and leverage AI smart notifications based on your real fitness habits.</p>", unsafe_allow_html=True)
    
    # Layer 1 / 2 / 3 Info Banner
    st.markdown("""
    <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.25); border-radius: 12px; padding: 12px 18px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #00F2FE;">Multi-Layer Notification System:</span>
        <span style="color: #CBD5E1; font-size: 0.88rem; margin-left: 8px;">
            Supports <b>Layer 1 (In-App)</b>, <b>Layer 2 (Browser Web Notification API)</b>, and <b>Layer 3 (Email SMTP & Push Webhooks)</b>.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Smart Alarm Widget
    smart_alarms = generate_smart_alarm_suggestions(user_id)
    if smart_alarms:
        st.subheader("🤖 AI Smart Alarm Recommendations")
        for sa in smart_alarms:
            st.markdown(f"""
            <div class="smart-alarm-box">
                <h4 style="margin: 0; color: #00F2FE;">{sa['title']}</h4>
                <p style="margin: 6px 0; color: #E2E8F0; font-size: 0.92rem;">{sa['message']}</p>
                <div style="font-weight: 600; color: #FFD166; font-size: 0.88rem; margin-bottom: 8px;">{sa['prompt']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            scol1, scol2 = st.columns([1, 4])
            with scol1:
                if st.button(f"⚡ {sa['action_title']}", key=f"btn_{sa['type']}", type="primary"):
                    add_reminder(
                        user_id=user_id,
                        title=sa['title'],
                        description=sa['message'],
                        date_str=datetime.now().strftime("%Y-%m-%d"),
                        time_str=sa.get('suggested_time', '18:00'),
                        repeat_type="Daily",
                        notification_type="In-App"
                    )
                    st.success("✅ Smart reminder scheduled successfully!")
                    st.rerun()
                    
    # Schedule New Reminder Form
    st.subheader("➕ Schedule New Reminder / Alarm")
    with st.container():
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            r_title = st.selectbox("Reminder Type", ["Workout Reminder", "Water Reminder", "Daily Goal Reminder", "Meal Reminder", "Sleep Reminder", "Custom Reminder"])
            r_custom_name = st.text_input("Custom Title (if custom)", placeholder="e.g. Evening Yoga mobility")
            final_title = r_custom_name if r_title == "Custom Reminder" and r_custom_name else r_title
        with rcol2:
            r_date = st.date_input("Date", value=datetime.now().date())
            r_time = st.time_input("Time", value=datetime.now().time())
        with rcol3:
            r_repeat = st.selectbox("Repeat", ["Once", "Daily", "Weekdays", "Weekly", "Custom"])
            r_channel = st.selectbox("Notification Channel", ["In-App", "Browser Notification", "Email", "Push Notification"])
            
        r_desc = st.text_input("Description / Notes", placeholder="Time to complete your workout session!")
        
        if st.button("⏰ Set Reminder", type="primary", use_container_width=True):
            add_reminder(
                user_id=user_id,
                title=final_title,
                description=r_desc,
                date_str=r_date.strftime("%Y-%m-%d"),
                time_str=r_time.strftime("%H:%M"),
                repeat_type=r_repeat,
                notification_type=r_channel
            )
            st.success(f"✅ Reminder '{final_title}' created successfully!")
            
            # Trigger Browser Notification JS if selected
            if r_channel == "Browser Notification":
                js_snippet = NotificationService.get_browser_notification_js(
                    f"Reminder Set: {final_title}",
                    f"Scheduled for {r_time.strftime('%I:%M %p')}"
                )
                st.components.v1.html(js_snippet, height=0)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Active Reminders Table & Management
    st.subheader("Scheduled Reminders in Database")
    rem_df = get_user_reminders(user_id)
    if not rem_df.empty:
        for _, r in rem_df.iterrows():
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">⏰ {r['title']}</div>
                    <div style="color: #94A3B8; font-size: 0.82rem;">{r['repeat_type']} at <b>{r['time']}</b> • Channel: <span style="color: #00F2FE;">{r['notification_type']}</span></div>
                    <div style="color: #64748B; font-size: 0.78rem;">{r['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            del_c1, del_c2 = st.columns([1, 5])
            with del_c1:
                if st.button(f"🗑️ Delete #{r['id']}", key=f"del_rem_{r['id']}"):
                    delete_reminder(r['id'], user_id)
                    st.success(f"Reminder #{r['id']} deleted.")
                    st.rerun()
    else:
        st.info("No reminders currently active.")

# =============================================================================
# 9. 🏆 ACHIEVEMENTS
# =============================================================================
elif nav_selection == "🏆 Achievements":
    st.title("🏆 Gamification & Achievement Badges")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Badges are unlocked automatically based on verified workout records and streaks.</p>", unsafe_allow_html=True)
    
    ach_list = get_user_achievements_status(user_id)
    unlocked_count = sum(1 for a in ach_list if a['is_unlocked'])
    total_count = len(ach_list)
    progress_pct = round((unlocked_count / total_count) * 100, 1)
    
    # Progress Bar Hero
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.write(f"### Overall Badge Mastery: **{unlocked_count} / {total_count}** ({progress_pct}%)")
    st.progress(progress_pct / 100.0)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Badges Grid
    cols = st.columns(3)
    for idx, ach in enumerate(ach_list):
        with cols[idx % 3]:
            card_class = "unlocked" if ach['is_unlocked'] else "locked"
            status_text = f"<span style='color: #FFD166; font-size: 0.75rem;'>✨ Unlocked on {ach['unlocked_at'][:10]}</span>" if ach['is_unlocked'] else "<span style='color: #64748B; font-size: 0.75rem;'>🔒 In Progress</span>"
            
            st.markdown(f"""
            <div class="achievement-card {card_class}">
                <div class="achievement-icon">{ach['icon']}</div>
                <div>
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">{ach['title']}</div>
                    <div style="color: #94A3B8; font-size: 0.8rem; margin: 2px 0 4px 0;">{ach['description']}</div>
                    {status_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# 10. 📄 REPORTS
# =============================================================================
elif nav_selection == "📄 Reports":
    st.title("📄 Professional Fitness Report Generator")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Generate and download a high-resolution, branded PDF fitness report using ReportLab.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.subheader("Report Contents Overview")
    st.write("Your generated PDF report includes:")
    st.write("• **User Profile & Fitness Parameters** (Name, Age, BMI category, Daily/Weekly goals)")
    st.write("• **Executive Fitness Summary** (Lifetime calories, total duration, average burn rate, streak)")
    st.write("• **Machine Learning & Deep Learning Performance** (MAE, RMSE, R² benchmark table)")
    st.write("• **Recent Workout Sessions Table** (Chronological real database records)")
    st.write("• **AI Habit Intelligence & Insights** (Week-over-week trends, optimal burn rates)")
    st.write("• **Gamification Badges** (List of unlocked achievements with timestamps)")
    
    if st.button("📄 Generate PDF Fitness Report", type="primary", use_container_width=True):
        with st.spinner("Compiling database records into styled ReportLab PDF..."):
            pdf_path = generate_pdf_report(user_id)
            st.session_state.last_pdf_path = pdf_path
            st.success(f"✅ PDF successfully compiled: `{os.path.basename(pdf_path)}`")
            
    if "last_pdf_path" in st.session_state and os.path.exists(st.session_state.last_pdf_path):
        with open(st.session_state.last_pdf_path, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="📥 Download PDF Fitness Report",
                data=pdf_bytes,
                file_name=os.path.basename(st.session_state.last_pdf_path),
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 11. 🤖 AI INSIGHTS
# =============================================================================
elif nav_selection == "🤖 AI Insights":
    st.title("🤖 AI Fitness & Habit Intelligence")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Evidence-based statistical analytics computed strictly from your actual workout history.</p>", unsafe_allow_html=True)
    
    insights = compute_user_insights(user_id)
    
    if insights.get('has_data', False):
        # AI Bullet Points
        st.subheader("Personalized AI Insights")
        for ins in insights['insights_list']:
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #00F2FE; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; color: #F8FAFC; font-size: 0.95rem;">
                {ins}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Advanced Analytics Metrics
        st.subheader("Statistical Habit Breakdown")
        icol1, icol2, icol3, icol4 = st.columns(4)
        icol1.metric("Consistency Rating", f"{insights['consistency_score']}%", help="Active workout days in past 14 days")
        icol2.metric("Favorite Exercise", insights['favorite_exercise'])
        icol3.metric("Top Calorie Burn Rate", f"{insights['top_efficient_rate']} kcal/min", help=f"Achieved during {insights['top_efficient_ex']}")
        icol4.metric("Habitual Workout Time", insights['habitual_time_str'])
        
        # Day of Week Frequency Chart
        workouts_df = get_all_workouts(user_id)
        if not workouts_df.empty:
            workouts_df['day_name'] = pd.to_datetime(workouts_df['date']).dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_counts = workouts_df['day_name'].value_counts().reindex(day_order, fill_value=0).reset_index()
            day_counts.columns = ['Day', 'Count']
            
            fig_day = px.bar(
                day_counts,
                x='Day',
                y='Count',
                title="Workout Frequency by Day of Week",
                color='Count',
                color_continuous_scale=[[0, '#00F2FE'], [1, '#FF6B6B']]
            )
            fig_day.update_layout(**PLOTLY_DARK_LAYOUT, height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.info("Log workouts in the system to enable AI habit intelligence.")

# =============================================================================
# 12. 👤 PROFILE
# =============================================================================
elif nav_selection == "👤 Profile":
    st.title("👤 Athlete Profile Management")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Manage your physiological profile, fitness goals, and basal metabolic parameters.</p>", unsafe_allow_html=True)
    
    pcol1, pcol2 = st.columns([1, 2])
    
    with pcol1:
        st.markdown("<div class='glass-container' style='text-align: center;'>", unsafe_allow_html=True)
        if user and user['profile_pic'] and os.path.exists(user['profile_pic']):
            st.image(user['profile_pic'], use_container_width=True)
        else:
            st.markdown("""
            <div style="width: 110px; height: 110px; border-radius: 50%; background: linear-gradient(135deg, #00F2FE 0%, #8B5CF6 100%); margin: 0 auto 14px auto; display: flex; align-items: center; justify-content: center; font-size: 48px;">
                👤
            </div>
            """, unsafe_allow_html=True)
            
        st.write(f"### {user['name'] if user else 'Athlete'}")
        st.caption(f"Member since {user['created_at'][:10] if user else '2026'}")
        
        # Calculate BMI & BMR
        if user:
            hm = user['height'] / 100.0
            bmi = round(user['weight'] / (hm * hm), 1)
            # Mifflin-St Jeor BMR
            if user['gender'] == 'male':
                bmr = int(round(10 * user['weight'] + 6.25 * user['height'] - 5 * user['age'] + 5))
            else:
                bmr = int(round(10 * user['weight'] + 6.25 * user['height'] - 5 * user['age'] - 161))
            st.metric("Body Mass Index (BMI)", f"{bmi}")
            st.metric("Basal Metabolic Rate (BMR)", f"{bmr} kcal/day")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with pcol2:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.subheader("Edit Profile Information")
        
        e_name = st.text_input("Full Name", value=user['name'] if user else "Alex Rivera")
        ec1, ec2 = st.columns(2)
        with ec1:
            e_age = st.number_input("Age (years)", min_value=15, max_value=90, value=int(user['age']) if user else 28)
            e_gender = st.selectbox("Gender", ["male", "female"], index=0 if user and user['gender'] == 'male' else 1)
            e_height = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=float(user['height']) if user else 178.0)
        with ec2:
            e_weight = st.number_input("Weight (kg)", min_value=35.0, max_value=200.0, value=float(user['weight']) if user else 75.0)
            e_daily = st.number_input("Daily Calorie Goal (kcal)", min_value=100.0, max_value=5000.0, value=float(user['daily_goal']) if user else 500.0)
            e_weekly = st.number_input("Weekly Calorie Goal (kcal)", min_value=500.0, max_value=35000.0, value=float(user['weekly_goal']) if user else 3500.0)
            
        e_goal = st.selectbox("Fitness Objective", ["Weight Loss & Endurance", "Muscle Gain & Hypertrophy", "Cardiovascular Health", "Athletic Performance", "General Wellness"])
        e_act = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"], index=2)
        
        if st.button("💾 Save Profile Changes", type="primary", use_container_width=True):
            update_user_profile(
                user_id, e_name, e_age, e_gender, e_height, e_weight, e_goal, e_act, e_daily, e_weekly
            )
            st.success("✅ Profile parameters successfully updated in database!")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 13. ⚙️ SETTINGS
# =============================================================================
elif nav_selection == "⚙️ Settings":
    st.title("⚙️ System Preferences & Integration Settings")
    st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Configure notification layers, preferred ML models, units, and security parameters.</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.subheader("🎨 User Interface & Model Preferences")
    
    scol1, scol2 = st.columns(2)
    with scol1:
        s_theme = st.selectbox("UI Theme Palette", ["Dark Neon", "Cyberpunk Glow", "Minimal Slate"], index=0)
        s_units = st.selectbox("Measurement Units", ["Metric (kg/cm)", "Imperial (lbs/in)"], index=0)
    with scol2:
        s_model = st.selectbox("Default Calorie Model", ["Auto (Best Model)", "Deep Neural Network (DNN)", "Gradient Boosting", "Random Forest", "Linear Regression"], index=0)
        s_sound = st.checkbox("Enable Interactive Audio Sound Effects", value=True)
        
    st.markdown("---")
    st.subheader("🔔 Modular Notification Channels (3-Layer Architecture)")
    
    # Layer 2: Browser Notification
    st.write("#### 🌐 Layer 2: Browser Web Notification API")
    if st.button("🔔 Test Native Browser Notification", type="secondary"):
        js_code = NotificationService.get_browser_notification_js(
            "🔥 Calories Burnt AI",
            "Browser notifications are active and working smoothly!"
        )
        st.components.v1.html(js_code, height=0)
        st.success("Sent browser notification test trigger.")
        
    # Layer 3: Email SMTP
    st.write("#### 📧 Layer 3: Email Alerts (SMTP)")
    s_email_addr = st.text_input("Recipient Email Address", value=settings.get('email_address', ''))
    if st.button("📨 Dispatch Test Email"):
        res = NotificationService.send_email_notification(
            s_email_addr,
            "Notification Test",
            "This is a test notification from your Calories Burnt Prediction AI Assistant."
        )
        if res['success']:
            st.success(res['message'])
        else:
            st.warning(res['message'])
            
    # Layer 3: Push Notification Webhook
    st.write("#### 📱 Layer 3: Push Notification Webhook")
    s_push_url = st.text_input("Push Webhook URL / Token", value=settings.get('push_token', ''), placeholder="https://api.pushover.net/1/messages.json or webhook URL")
    if st.button("📱 Dispatch Test Push Notification"):
        res = NotificationService.send_push_notification("Push Test", "Workout reminder triggered!", s_push_url)
        if res['success']:
            st.success(res['message'])
        else:
            st.warning(res['message'])
            
    st.markdown("---")
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        update_user_settings(
            user_id=user_id,
            theme=s_theme,
            preferred_model=s_model,
            sound_enabled=1 if s_sound else 0,
            browser_notifications=1,
            email_notifications=1 if s_email_addr else 0,
            push_notifications=1 if s_push_url else 0,
            email_address=s_email_addr,
            push_token=s_push_url,
            units=s_units
        )
        st.success("✅ Settings updated successfully!")
    st.markdown("</div>", unsafe_allow_html=True)
