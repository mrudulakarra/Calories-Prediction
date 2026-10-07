"""
Achievements & Gamification Engine for Calorie Burnt Prediction AI
Evaluates user activity and unlocks achievement badges automatically based on database history.
"""

import os
import sys
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database import get_connection, get_dashboard_metrics

ALL_ACHIEVEMENTS = [
    {
        "badge_id": "first_workout",
        "title": "First Workout",
        "description": "Completed and logged your very first workout session!",
        "icon": "🥇",
        "category": "Milestone"
    },
    {
        "badge_id": "cal_1000",
        "title": "1,000 Calories Burned",
        "description": "Surpassed 1,000 cumulative calories burned.",
        "icon": "🔥",
        "category": "Calorie"
    },
    {
        "badge_id": "cal_5000",
        "title": "5,000 Calories Burned",
        "description": "Surpassed 5,000 cumulative calories burned.",
        "icon": "🔥",
        "category": "Calorie"
    },
    {
        "badge_id": "cal_10000",
        "title": "10,000 Calories Burned",
        "description": "Surpassed 10,000 cumulative calories burned.",
        "icon": "👑",
        "category": "Calorie"
    },
    {
        "badge_id": "workouts_10",
        "title": "10 Workouts Club",
        "description": "Completed 10 total fitness sessions.",
        "icon": "🏃",
        "category": "Workout Count"
    },
    {
        "badge_id": "workouts_25",
        "title": "25 Workouts Veteran",
        "description": "Completed 25 total fitness sessions.",
        "icon": "🎖️",
        "category": "Workout Count"
    },
    {
        "badge_id": "streak_7",
        "title": "7 Day Streak",
        "description": "Maintained an active workout streak for 7 consecutive days.",
        "icon": "⚡",
        "category": "Streak"
    },
    {
        "badge_id": "streak_30",
        "title": "30 Day Streak",
        "description": "Maintained an active workout streak for 30 consecutive days.",
        "icon": "🌟",
        "category": "Streak"
    },
    {
        "badge_id": "goal_crusher",
        "title": "Goal Crusher",
        "description": "Crushed your daily calorie target on 5 distinct days.",
        "icon": "🎯",
        "category": "Goals"
    },
    {
        "badge_id": "consistency_champ",
        "title": "Consistency Champion",
        "description": "Completed 4 or more workouts within the past 7 days.",
        "icon": "💪",
        "category": "Consistency"
    },
    {
        "badge_id": "endurance_master",
        "title": "Endurance Master",
        "description": "Logged a single workout lasting 50 minutes or more.",
        "icon": "⏱️",
        "category": "Endurance"
    },
    {
        "badge_id": "hiit_hero",
        "title": "High Intensity Hero",
        "description": "Completed a workout in the Maximum / Peak intensity zone.",
        "icon": "🚀",
        "category": "Intensity"
    }
]

def check_and_unlock_achievements(user_id=1):
    """
    Evaluates real workout and streak data in the database,
    unlocking any newly eligible achievements.
    Returns list of newly unlocked achievement dictionaries.
    """
    conn = get_connection()
    cur = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get currently unlocked badge IDs
    cur.execute("SELECT badge_id FROM achievements WHERE user_id = ?", (user_id,))
    already_unlocked = {row[0] for row in cur.fetchall()}
    
    # Fetch user data & workout stats
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    daily_goal = user['daily_goal'] if user else 500.0
    
    df_workouts = pd.read_sql_query("SELECT * FROM workouts WHERE user_id = ?", conn, params=(user_id,))
    metrics = get_dashboard_metrics(user_id)
    
    total_workouts = len(df_workouts)
    total_calories = df_workouts['calories'].sum() if not df_workouts.empty else 0
    max_duration = df_workouts['duration'].max() if not df_workouts.empty else 0
    has_max_intensity = False
    if not df_workouts.empty and 'intensity' in df_workouts.columns:
        has_max_intensity = (df_workouts['intensity'] == 'Maximum').any()
        
    # Count days meeting daily goal
    days_met_goal = 0
    if not df_workouts.empty:
        daily_sums = df_workouts.groupby('date')['calories'].sum()
        days_met_goal = (daily_sums >= daily_goal).sum()
        
    streak = metrics.get('current_streak', 0)
    
    # Workouts in last 7 days
    workouts_last_7_days = 0
    if not df_workouts.empty:
        df_workouts['dt'] = pd.to_datetime(df_workouts['date'])
        seven_days_ago = pd.to_datetime(datetime.now().date()) - pd.Timedelta(days=7)
        workouts_last_7_days = (df_workouts['dt'] >= seven_days_ago).sum()
        
    newly_unlocked = []
    
    for ach in ALL_ACHIEVEMENTS:
        b_id = ach['badge_id']
        if b_id in already_unlocked:
            continue
            
        qualifies = False
        
        if b_id == "first_workout" and total_workouts >= 1:
            qualifies = True
        elif b_id == "cal_1000" and total_calories >= 1000:
            qualifies = True
        elif b_id == "cal_5000" and total_calories >= 5000:
            qualifies = True
        elif b_id == "cal_10000" and total_calories >= 10000:
            qualifies = True
        elif b_id == "workouts_10" and total_workouts >= 10:
            qualifies = True
        elif b_id == "workouts_25" and total_workouts >= 25:
            qualifies = True
        elif b_id == "streak_7" and streak >= 7:
            qualifies = True
        elif b_id == "streak_30" and streak >= 30:
            qualifies = True
        elif b_id == "goal_crusher" and days_met_goal >= 5:
            qualifies = True
        elif b_id == "consistency_champ" and workouts_last_7_days >= 4:
            qualifies = True
        elif b_id == "endurance_master" and max_duration >= 50:
            qualifies = True
        elif b_id == "hiit_hero" and has_max_intensity:
            qualifies = True
            
        if qualifies:
            cur.execute("""
            INSERT OR IGNORE INTO achievements (user_id, badge_id, title, description, icon, unlocked_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, b_id, ach['title'], ach['description'], ach['icon'], now_str))
            newly_unlocked.append(ach)
            
    conn.commit()
    conn.close()
    return newly_unlocked

def get_user_achievements_status(user_id=1):
    """
    Returns full list of all achievements with their unlocked status and timestamp.
    """
    check_and_unlock_achievements(user_id)
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT badge_id, unlocked_at FROM achievements WHERE user_id = ?", (user_id,))
    unlocked_dict = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    
    result = []
    for ach in ALL_ACHIEVEMENTS:
        b_id = ach['badge_id']
        is_unlocked = b_id in unlocked_dict
        result.append({
            **ach,
            "is_unlocked": is_unlocked,
            "unlocked_at": unlocked_dict.get(b_id, None)
        })
        
    return result

if __name__ == "__main__":
    status = get_user_achievements_status(1)
    unlocked_count = sum(1 for a in status if a['is_unlocked'])
    print(f"Total Achievements: {len(status)}, Unlocked: {unlocked_count}")
