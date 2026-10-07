"""
Alarms and Smart AI Reminder Scheduling System
Manages recurring timers, checks active reminders, and generates real-time smart prompts.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database import get_connection, get_dashboard_metrics
from src.insights import compute_user_insights

def get_upcoming_alarms(user_id=1):
    """
    Returns list of upcoming active reminders for today and this week.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM reminders WHERE user_id = ? AND enabled = 1 ORDER BY time ASC",
        conn,
        params=(user_id,)
    )
    conn.close()
    return df

def generate_smart_alarm_suggestions(user_id=1):
    """
    Analyzes current time, today's calorie progress, and historical workout habits
    to generate intelligent suggestions.
    """
    metrics = get_dashboard_metrics(user_id)
    insights = compute_user_insights(user_id)
    
    now = datetime.now()
    current_hour = now.hour
    
    today_cals = metrics.get('today_calories', 0)
    daily_goal = metrics.get('daily_goal', 500)
    goal_pct = metrics.get('daily_goal_pct', 0)
    streak = metrics.get('current_streak', 0)
    
    habitual_hour = insights.get('habitual_workout_hour', 18)
    habitual_time_str = insights.get('habitual_time_str', '6:00 PM')
    
    suggestions = []
    
    # 1. Goal completion check (if past 14:00 and goal < 60%)
    if goal_pct < 60 and current_hour >= 14:
        deficit = int(round(daily_goal - today_cals))
        suggestions.append({
            "type": "calorie_deficit",
            "title": "🎯 Daily Calorie Goal Alert",
            "message": f"Your daily calorie goal is currently {goal_pct}% complete ({int(today_cals)} / {int(daily_goal)} kcal). You need ~{deficit} kcal more today.",
            "prompt": "Would you like to schedule a 30-minute workout session?",
            "action_title": "Schedule 30-Min Workout",
            "suggested_duration": 30,
            "suggested_exercise": insights.get('favorite_exercise', 'Running')
        })
        
    # 2. Habitual workout time reminder prompt
    if current_hour < habitual_hour:
        suggestions.append({
            "type": "habitual_time",
            "title": "💡 Habitual Workout Reminder",
            "message": f"Based on your activity logs, you usually exercise around {habitual_time_str}.",
            "prompt": f"Set today's workout reminder for {habitual_time_str}?",
            "action_title": f"Set Reminder for {habitual_time_str}",
            "suggested_time": f"{habitual_hour:02d}:00",
            "suggested_exercise": insights.get('favorite_exercise', 'Running')
        })
        
    # 3. Streak Protection
    if streak >= 3 and today_cals == 0 and current_hour >= 16:
        suggestions.append({
            "type": "streak_protection",
            "title": "🔥 Streak at Risk!",
            "message": f"You are currently on a {streak}-day workout streak! Don't break the chain today.",
            "prompt": "Even a quick 15-minute high-intensity workout will keep your streak alive.",
            "action_title": "Quick 15-Min Session",
            "suggested_duration": 15,
            "suggested_exercise": "HIIT"
        })
        
    return suggestions

if __name__ == "__main__":
    suggs = generate_smart_alarm_suggestions(1)
    print("Smart Alarm Suggestions:")
    for s in suggs:
        print(f" - {s['title']}: {s['message']}")
