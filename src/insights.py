"""
AI Insights & Advanced Analytics Engine
Computes evidence-based statistics, workout trends, consistency metrics,
and personalized exercise insights from actual database records.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database import get_connection, get_dashboard_metrics

def compute_user_insights(user_id=1):
    """
    Analyzes historical workout records to generate granular metrics and AI insights.
    """
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM workouts WHERE user_id = ? ORDER BY date ASC, time ASC", conn, params=(user_id,))
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    
    if df.empty:
        return {
            "has_data": False,
            "insights_list": ["No workouts recorded yet. Start logging your workouts to view personalized AI insights!"],
            "total_workouts": 0,
            "total_calories": 0,
            "avg_calories": 0,
            "avg_duration": 0,
            "favorite_exercise": "N/A",
            "habitual_workout_hour": 18,
            "consistency_score": 0
        }
        
    df['dt'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
    df['day_name'] = df['dt'].dt.day_name()
    
    total_workouts = len(df)
    total_calories = float(df['calories'].sum())
    avg_calories = float(df['calories'].mean())
    avg_duration = float(df['duration'].mean())
    avg_heart_rate = float(df['heart_rate'].mean())
    
    # Most frequent exercise
    fav_exercise = df['exercise'].value_counts().index[0]
    fav_exercise_count = df['exercise'].value_counts().iloc[0]
    fav_exercise_pct = round((fav_exercise_count / total_workouts) * 100, 1)
    
    # Best calorie burning exercise per minute
    df['cal_per_min'] = df['calories'] / df['duration'].clip(lower=1)
    cal_burn_by_ex = df.groupby('exercise')['cal_per_min'].mean().to_dict()
    top_efficient_ex = max(cal_burn_by_ex.items(), key=lambda x: x[1])[0]
    top_efficient_rate = round(cal_burn_by_ex[top_efficient_ex], 1)
    
    # Habitual workout time
    valid_hours = df['hour'].dropna()
    habitual_hour = int(valid_hours.mode()[0]) if not valid_hours.empty else 18
    am_pm_str = f"{habitual_hour % 12 or 12}:00 {'PM' if habitual_hour >= 12 else 'AM'}"
    
    # Week-over-week calorie change
    now_date = datetime.now().date()
    curr_week_start = pd.to_datetime(now_date - timedelta(days=6))
    prev_week_start = pd.to_datetime(now_date - timedelta(days=13))
    prev_week_end = pd.to_datetime(now_date - timedelta(days=7))
    
    curr_week_cals = df[df['dt'] >= curr_week_start]['calories'].sum()
    prev_week_cals = df[(df['dt'] >= prev_week_start) & (df['dt'] <= prev_week_end)]['calories'].sum()
    
    if prev_week_cals > 0:
        wow_change = round(((curr_week_cals - prev_week_cals) / prev_week_cals) * 100.0, 1)
    else:
        wow_change = 0.0
        
    # Day-of-week frequency
    day_counts = df['day_name'].value_counts()
    peak_day = day_counts.index[0]
    
    # Consistency Score (Percentage of days with workouts in the last 14 days)
    fourteen_days_ago = pd.to_datetime(now_date - timedelta(days=13))
    recent_active_days = df[df['dt'] >= fourteen_days_ago]['dt'].dt.date.nunique()
    consistency_score = min(round((recent_active_days / 14.0) * 100.0, 1), 100.0)
    
    # Dashboard metrics (for streak and goal)
    dash = get_dashboard_metrics(user_id)
    streak = dash.get('current_streak', 0)
    weekly_goal_pct = dash.get('weekly_goal_pct', 0)
    
    # Generate contextual AI insights
    insights_list = []
    
    if wow_change > 0:
        insights_list.append(f"📈 Your weekly calorie burn increased by {wow_change}% compared to the prior week.")
    elif wow_change < 0:
        insights_list.append(f"📉 Your weekly calorie output is {abs(wow_change)}% lower than last week. Consider scheduling an extra session.")
    else:
        insights_list.append(f"⚖️ Your weekly calorie burn is consistent with last week ({round(curr_week_cals, 0)} kcal).")
        
    insights_list.append(f"🏃 Your most frequent workout is {fav_exercise} ({fav_exercise_pct}% of total sessions).")
    insights_list.append(f"⏱️ Your average workout duration is {round(avg_duration, 1)} minutes with an average burn of {round(avg_calories, 1)} kcal.")
    insights_list.append(f"⚡ {top_efficient_ex} yields your highest calorie burn efficiency at ~{top_efficient_rate} kcal/minute.")
    
    if streak >= 3:
        insights_list.append(f"🔥 Active streak: You have completed {streak} consecutive workout days!")
        
    insights_list.append(f"🎯 You are currently {weekly_goal_pct}% toward your weekly calorie goal ({round(curr_week_cals, 0)} / {dash.get('weekly_goal', 3500)} kcal).")
    insights_list.append(f"💡 You most frequently exercise around {am_pm_str} on {peak_day}s.")
    
    return {
        "has_data": True,
        "insights_list": insights_list,
        "total_workouts": total_workouts,
        "total_calories": round(total_calories, 1),
        "avg_calories": round(avg_calories, 1),
        "avg_duration": round(avg_duration, 1),
        "avg_heart_rate": round(avg_heart_rate, 1),
        "favorite_exercise": fav_exercise,
        "fav_exercise_count": fav_exercise_count,
        "top_efficient_ex": top_efficient_ex,
        "top_efficient_rate": top_efficient_rate,
        "habitual_workout_hour": habitual_hour,
        "habitual_time_str": am_pm_str,
        "peak_day": peak_day,
        "wow_change": wow_change,
        "curr_week_cals": round(curr_week_cals, 1),
        "prev_week_cals": round(prev_week_cals, 1),
        "consistency_score": consistency_score,
        "recent_active_days": recent_active_days
    }

if __name__ == "__main__":
    res = compute_user_insights(1)
    print("User AI Insights:")
    for ins in res['insights_list']:
        print(f" - {ins}")
