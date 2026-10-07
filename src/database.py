"""
SQLite Database Layer for Calorie Burnt Prediction AI
Handles persistent storage for users, workouts, goals, achievements, reminders, notifications, and progress photos.
"""

import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

DB_PATH = os.path.join("database", "fitness.db")

def get_connection(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DB_PATH):
    """Initializes all database tables and seeds default user if none exists."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    
    # 1. Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        height REAL NOT NULL,
        weight REAL NOT NULL,
        fitness_goal TEXT NOT NULL,
        activity_level TEXT NOT NULL,
        daily_goal REAL NOT NULL,
        weekly_goal REAL NOT NULL,
        profile_pic TEXT,
        created_at TEXT NOT NULL
    )
    """)
    
    # 2. Workouts table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        exercise TEXT NOT NULL,
        duration INTEGER NOT NULL,
        heart_rate INTEGER NOT NULL,
        body_temp REAL NOT NULL,
        calories REAL NOT NULL,
        intensity TEXT NOT NULL,
        input_source TEXT NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 3. Goals table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        goal_type TEXT NOT NULL,
        target_value REAL NOT NULL,
        period TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 4. Achievements table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS achievements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        badge_id TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        icon TEXT NOT NULL,
        unlocked_at TEXT NOT NULL,
        UNIQUE(user_id, badge_id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 5. Reminders table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        description TEXT,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        repeat_type TEXT NOT NULL,
        notification_type TEXT NOT NULL,
        enabled INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 6. Notifications table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS notifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        notification_type TEXT NOT NULL,
        status TEXT NOT NULL,
        scheduled_at TEXT NOT NULL,
        sent_at TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 7. Progress Photos table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS progress_photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        photo_type TEXT NOT NULL,
        file_path TEXT NOT NULL,
        date TEXT NOT NULL,
        weight REAL,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    # 8. Settings table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        theme TEXT DEFAULT 'Dark Neon',
        preferred_model TEXT DEFAULT 'Auto (Best Model)',
        sound_enabled INTEGER DEFAULT 1,
        browser_notifications INTEGER DEFAULT 1,
        email_notifications INTEGER DEFAULT 0,
        push_notifications INTEGER DEFAULT 0,
        email_address TEXT DEFAULT '',
        push_token TEXT DEFAULT '',
        units TEXT DEFAULT 'Metric (kg/cm)',
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)
    
    conn.commit()
    
    # Check if default user exists, if not seed initial demo data
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        seed_initial_data(conn)
        
    conn.close()

def seed_initial_data(conn):
    """Seeds default user, initial goals, reminders, and realistic 14-day workout history."""
    cur = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Default User
    cur.execute("""
    INSERT INTO users (name, age, gender, height, weight, fitness_goal, activity_level, daily_goal, weekly_goal, profile_pic, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("Alex Rivera", 28, "male", 178.0, 75.0, "Weight Loss & Endurance", "Moderately Active", 500.0, 3500.0, "", now_str))
    user_id = cur.lastrowid
    
    # Default Settings
    cur.execute("""
    INSERT INTO settings (user_id, theme, preferred_model, sound_enabled, browser_notifications, email_notifications, push_notifications, email_address, push_token, units)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, "Dark Neon", "Auto (Best Model)", 1, 1, 0, 0, "alex.rivera@example.com", "", "Metric (kg/cm)"))
    
    # Default Goals
    goals_data = [
        (user_id, "Daily Calorie Goal", 500.0, "daily", (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"), None, 1, now_str),
        (user_id, "Weekly Calorie Goal", 3500.0, "weekly", (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"), None, 1, now_str),
        (user_id, "Daily Duration Goal", 45.0, "daily", (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"), None, 1, now_str),
        (user_id, "Weekly Frequency Goal", 5.0, "weekly", (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"), None, 1, now_str)
    ]
    cur.executemany("""
    INSERT INTO goals (user_id, goal_type, target_value, period, start_date, end_date, is_active, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, goals_data)
    
    # Default Reminders
    reminders_data = [
        (user_id, "Evening Workout Reminder", "Time for your daily fitness session!", datetime.now().strftime("%Y-%m-%d"), "18:00", "Daily", "In-App", 1, now_str),
        (user_id, "Hydration Check", "Drink at least 500ml water to stay hydrated", datetime.now().strftime("%Y-%m-%d"), "14:30", "Daily", "In-App", 1, now_str),
        (user_id, "Daily Goal Check", "Review daily calorie progress before evening", datetime.now().strftime("%Y-%m-%d"), "20:00", "Daily", "In-App", 1, now_str)
    ]
    cur.executemany("""
    INSERT INTO reminders (user_id, title, description, date, time, repeat_type, notification_type, enabled, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, reminders_data)
    
    # Seed 14 days of realistic workout history leading up to today
    workouts_seed = [
        (13, "Running", 40, 155, 38.8, 460.5, "Vigorous", "Manual", "Great morning outdoor run"),
        (12, "HIIT", 35, 162, 39.1, 485.0, "Maximum", "Manual", "High intensity interval training"),
        (11, "Cycling", 45, 142, 38.3, 410.0, "Moderate", "Manual", "Stationary bike workout"),
        (10, "Strength Training", 50, 138, 38.1, 390.0, "Moderate", "Manual", "Upper body push & pull"),
        (9, "Swimming", 40, 148, 38.2, 430.0, "Vigorous", "Manual", "Freestyle laps"),
        (8, "Running", 45, 158, 38.9, 520.0, "Vigorous", "Manual", "Tempo run"),
        (7, "Yoga", 50, 105, 37.4, 195.0, "Low", "Manual", "Recovery and mobility flow"),
        (6, "Gym", 55, 144, 38.4, 470.0, "Moderate", "Manual", "Leg day squats and lunges"),
        (5, "Running", 35, 154, 38.7, 415.0, "Vigorous", "Manual", "Evening trail jog"),
        (4, "HIIT", 30, 165, 39.2, 450.0, "Maximum", "Manual", "Full body circuit"),
        (3, "Cycling", 50, 145, 38.5, 480.0, "Moderate", "Manual", "Outdoor road cycling"),
        (2, "Strength Training", 45, 135, 38.0, 360.0, "Moderate", "Manual", "Core & lower body session"),
        (1, "Running", 42, 156, 38.8, 490.0, "Vigorous", "Manual", "Interval speed session"),
        (0, "HIIT", 40, 160, 39.0, 486.0, "Maximum", "Manual", "Today's dynamic cardio circuit")
    ]
    
    for days_ago, ex, dur, hr, temp, cal, intens, src, notes in workouts_seed:
        w_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        w_time = "18:15:00"
        cur.execute("""
        INSERT INTO workouts (user_id, date, time, exercise, duration, heart_rate, body_temp, calories, intensity, input_source, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, w_date, w_time, ex, dur, hr, temp, cal, intens, src, notes, now_str))
        
    conn.commit()

# --- User Functions ---
def get_user_profile(user_id=1):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

def update_user_profile(user_id, name, age, gender, height, weight, fitness_goal, activity_level, daily_goal, weekly_goal, profile_pic=""):
    conn = get_connection()
    if profile_pic:
        conn.execute("""
        UPDATE users SET name=?, age=?, gender=?, height=?, weight=?, fitness_goal=?, activity_level=?, daily_goal=?, weekly_goal=?, profile_pic=?
        WHERE id=?
        """, (name, age, gender, height, weight, fitness_goal, activity_level, daily_goal, weekly_goal, profile_pic, user_id))
    else:
        conn.execute("""
        UPDATE users SET name=?, age=?, gender=?, height=?, weight=?, fitness_goal=?, activity_level=?, daily_goal=?, weekly_goal=?
        WHERE id=?
        """, (name, age, gender, height, weight, fitness_goal, activity_level, daily_goal, weekly_goal, user_id))
    conn.commit()
    conn.close()

# --- Workout Functions ---
def log_workout(user_id, exercise, duration, heart_rate, body_temp, calories, intensity, input_source="Manual", notes="", workout_date=None, workout_time=None):
    conn = get_connection()
    now = datetime.now()
    w_date = workout_date if workout_date else now.strftime("%Y-%m-%d")
    w_time = workout_time if workout_time else now.strftime("%H:%M:%S")
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO workouts (user_id, date, time, exercise, duration, heart_rate, body_temp, calories, intensity, input_source, notes, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, w_date, w_time, exercise, duration, heart_rate, body_temp, calories, intensity, input_source, notes, now_str))
    
    workout_id = cur.lastrowid
    conn.commit()
    conn.close()
    
    return workout_id

def get_all_workouts(user_id=1, limit=None):
    conn = get_connection()
    query = "SELECT * FROM workouts WHERE user_id = ? ORDER BY date DESC, time DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df

def delete_workout(workout_id, user_id=1):
    conn = get_connection()
    conn.execute("DELETE FROM workouts WHERE id = ? AND user_id = ?", (workout_id, user_id))
    conn.commit()
    conn.close()

# --- Dashboard & Statistics Queries ---
def get_dashboard_metrics(user_id=1):
    """
    Computes real metrics directly from SQLite database:
    - Calories burned today
    - Workout duration today
    - Workouts completed today
    - Daily goal completion %
    - Current consecutive active streak
    - Weekly total calories (past 7 days)
    """
    conn = get_connection()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # User Daily Goal
    user = conn.execute("SELECT daily_goal, weekly_goal FROM users WHERE id = ?", (user_id,)).fetchone()
    daily_goal = user['daily_goal'] if user else 500.0
    weekly_goal = user['weekly_goal'] if user else 3500.0
    
    # Today's stats
    today_row = conn.execute("""
    SELECT 
        COALESCE(SUM(calories), 0) as total_cals,
        COALESCE(SUM(duration), 0) as total_dur,
        COUNT(*) as workout_count
    FROM workouts 
    WHERE user_id = ? AND date = ?
    """, (user_id, today_str)).fetchone()
    
    today_cals = float(today_row['total_cals'])
    today_dur = int(today_row['total_dur'])
    today_workouts = int(today_row['workout_count'])
    daily_goal_pct = round(min((today_cals / max(daily_goal, 1.0)) * 100.0, 100.0), 1)
    
    # Past 7 days calories
    seven_days_ago = (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
    weekly_row = conn.execute("""
    SELECT COALESCE(SUM(calories), 0) as week_cals
    FROM workouts 
    WHERE user_id = ? AND date >= ?
    """, (user_id, seven_days_ago)).fetchone()
    weekly_cals = float(weekly_row['week_cals'])
    weekly_goal_pct = round(min((weekly_cals / max(weekly_goal, 1.0)) * 100.0, 100.0), 1)
    
    # Calculate streak from actual workout dates
    workout_dates = [row[0] for row in conn.execute(
        "SELECT DISTINCT date FROM workouts WHERE user_id = ? ORDER BY date DESC", (user_id,)
    ).fetchall()]
    
    streak = 0
    check_date = datetime.now().date()
    
    # Check if today has workout; if not, check if yesterday had one to maintain streak
    today_date_str = check_date.strftime("%Y-%m-%d")
    if today_date_str in workout_dates:
        streak += 1
        check_date = check_date - timedelta(days=1)
    else:
        yesterday_str = (check_date - timedelta(days=1)).strftime("%Y-%m-%d")
        if yesterday_str in workout_dates:
            check_date = check_date - timedelta(days=1)
        else:
            check_date = None
            
    while check_date is not None:
        d_str = check_date.strftime("%Y-%m-%d")
        if d_str in workout_dates:
            streak += 1
            check_date = check_date - timedelta(days=1)
        else:
            break
            
    conn.close()
    
    return {
        "today_calories": round(today_cals, 1),
        "today_duration": today_dur,
        "today_workouts": today_workouts,
        "daily_goal": daily_goal,
        "daily_goal_pct": daily_goal_pct,
        "weekly_calories": round(weekly_cals, 1),
        "weekly_goal": weekly_goal,
        "weekly_goal_pct": weekly_goal_pct,
        "current_streak": streak
    }

# --- Goals Functions ---
def get_user_goals(user_id=1):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM goals WHERE user_id = ? ORDER BY id ASC", conn, params=(user_id,))
    conn.close()
    return df

def save_user_goal(user_id, goal_type, target_value, period="daily"):
    conn = get_connection()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_str = datetime.now().strftime("%Y-%m-%d")
    conn.execute("""
    INSERT INTO goals (user_id, goal_type, target_value, period, start_date, is_active, created_at)
    VALUES (?, ?, ?, ?, ?, 1, ?)
    """, (user_id, goal_type, target_value, period, start_str, now_str))
    conn.commit()
    conn.close()

def delete_user_goal(goal_id, user_id=1):
    conn = get_connection()
    conn.execute("DELETE FROM goals WHERE id = ? AND user_id = ?", (goal_id, user_id))
    conn.commit()
    conn.close()

# --- Reminders & Alarms Functions ---
def get_user_reminders(user_id=1):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM reminders WHERE user_id = ? ORDER BY time ASC", conn, params=(user_id,))
    conn.close()
    return df

def add_reminder(user_id, title, description, date_str, time_str, repeat_type="Daily", notification_type="In-App"):
    conn = get_connection()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO reminders (user_id, title, description, date, time, repeat_type, notification_type, enabled, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
    """, (user_id, title, description, date_str, time_str, repeat_type, notification_type, now_str))
    r_id = cur.lastrowid
    conn.commit()
    conn.close()
    return r_id

def toggle_reminder(reminder_id, enabled_status, user_id=1):
    conn = get_connection()
    conn.execute("UPDATE reminders SET enabled = ? WHERE id = ? AND user_id = ?", (int(enabled_status), reminder_id, user_id))
    conn.commit()
    conn.close()

def delete_reminder(reminder_id, user_id=1):
    conn = get_connection()
    conn.execute("DELETE FROM reminders WHERE id = ? AND user_id = ?", (reminder_id, user_id))
    conn.commit()
    conn.close()

# --- Progress Photos Functions ---
def save_progress_photo(user_id, photo_type, file_path, date_str, weight=None, notes=""):
    conn = get_connection()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO progress_photos (user_id, photo_type, file_path, date, weight, notes, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, photo_type, file_path, date_str, weight, notes, now_str))
    conn.commit()
    conn.close()

def get_progress_photos(user_id=1):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM progress_photos WHERE user_id = ? ORDER BY date DESC", conn, params=(user_id,))
    conn.close()
    return df

# --- Settings Functions ---
def get_user_settings(user_id=1):
    conn = get_connection()
    row = conn.execute("SELECT * FROM settings WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {
        "theme": "Dark Neon",
        "preferred_model": "Auto (Best Model)",
        "sound_enabled": 1,
        "browser_notifications": 1,
        "email_notifications": 0,
        "push_notifications": 0,
        "email_address": "",
        "push_token": "",
        "units": "Metric (kg/cm)"
    }

def update_user_settings(user_id, theme, preferred_model, sound_enabled, browser_notifications, email_notifications, push_notifications, email_address="", push_token="", units="Metric (kg/cm)"):
    conn = get_connection()
    conn.execute("""
    UPDATE settings 
    SET theme=?, preferred_model=?, sound_enabled=?, browser_notifications=?, email_notifications=?, push_notifications=?, email_address=?, push_token=?, units=?
    WHERE user_id=?
    """, (theme, preferred_model, sound_enabled, browser_notifications, email_notifications, push_notifications, email_address, push_token, units, user_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    metrics = get_dashboard_metrics(1)
    print("Database Initialized. Seed Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
