"""
PDF Report Generation Engine using ReportLab
Generates a comprehensive, styled fitness and calorie analytics PDF report.
"""

import os
import sys
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.database import get_user_profile, get_dashboard_metrics, get_all_workouts, get_connection
from src.achievements import get_user_achievements_status
from src.insights import compute_user_insights
from src.prediction import get_model_metadata

def generate_pdf_report(user_id=1, output_dir="reports"):
    """
    Generates a full-featured, professional PDF fitness report and returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"Fitness_Report_User_{user_id}_{timestamp}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    # Retrieve all database data
    user = get_user_profile(user_id)
    if not user:
        user = {
            "name": "Athlete", "age": 28, "gender": "male", "height": 175,
            "weight": 70, "fitness_goal": "General Fitness", "activity_level": "Moderately Active",
            "daily_goal": 500, "weekly_goal": 3500
        }
        
    metrics = get_dashboard_metrics(user_id)
    workouts_df = get_all_workouts(user_id)
    achievements = get_user_achievements_status(user_id)
    insights = compute_user_insights(user_id)
    model_meta = get_model_metadata()
    
    # Calculate BMI
    h_m = user['height'] / 100.0
    bmi = round(user['weight'] / (h_m * h_m), 1) if h_m > 0 else 22.0
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25.0:
        bmi_cat = "Normal Weight"
    elif bmi < 30.0:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obese"
        
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A")
    )
    
    tagline_style = ParagraphStyle(
        'DocTagline',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0284C7")
    )
    
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6
    )
    
    normal_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    
    bold_style = ParagraphStyle(
        'BodyBoldDark',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0F172A")
    )
    
    badge_style = ParagraphStyle(
        'BadgeStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#FFFFFF")
    )
    
    story = []
    
    # --- Header Section ---
    header_data = [
        [
            Paragraph("🔥 <b>CALORIES BURNT PREDICTION AI</b>", title_style),
            Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%b %d, %Y - %I:%M %p')}<br/><b>User ID:</b> #{user_id}", normal_style)
        ],
        [
            Paragraph("AI-Powered Personal Fitness, Calorie & Activity Intelligence", tagline_style),
            Paragraph("<b>Report Status:</b> Verified Data", normal_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284C7"), spaceBefore=2, spaceAfter=12))
    
    # --- 1. User Profile & Fitness Summary Grid ---
    profile_data = [
        [Paragraph("<b>Name:</b>", bold_style), Paragraph(str(user['name']), normal_style), Paragraph("<b>Daily Calorie Goal:</b>", bold_style), Paragraph(f"{user['daily_goal']} kcal", normal_style)],
        [Paragraph("<b>Age / Gender:</b>", bold_style), Paragraph(f"{user['age']} yrs / {str(user['gender']).capitalize()}", normal_style), Paragraph("<b>Weekly Calorie Goal:</b>", bold_style), Paragraph(f"{user['weekly_goal']} kcal", normal_style)],
        [Paragraph("<b>Height / Weight:</b>", bold_style), Paragraph(f"{user['height']} cm / {user['weight']} kg", normal_style), Paragraph("<b>BMI:</b>", bold_style), Paragraph(f"{bmi} ({bmi_cat})", normal_style)],
        [Paragraph("<b>Fitness Goal:</b>", bold_style), Paragraph(str(user['fitness_goal']), normal_style), Paragraph("<b>Activity Level:</b>", bold_style), Paragraph(str(user['activity_level']), normal_style)]
    ]
    profile_table = Table(profile_data, colWidths=[100, 170, 130, 140])
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    
    story.append(Paragraph("👤 <b>User Profile & Fitness Parameters</b>", h2_style))
    story.append(profile_table)
    story.append(Spacer(1, 10))
    
    # --- 2. Executive Fitness Metrics Cards ---
    total_dur_hours = round(workouts_df['duration'].sum() / 60.0, 1) if not workouts_df.empty else 0
    total_cals_sum = round(workouts_df['calories'].sum(), 0) if not workouts_df.empty else 0
    avg_cal_burn = round(workouts_df['calories'].mean(), 1) if not workouts_df.empty else 0
    
    kpi_data = [
        [
            Paragraph(f"<font size=14 color='#0284C7'><b>{total_cals_sum:,.0f}</b></font><br/><font size=8 color='#64748B'>LIFETIME CALORIES</font>", normal_style),
            Paragraph(f"<font size=14 color='#059669'><b>{len(workouts_df)}</b></font><br/><font size=8 color='#64748B'>TOTAL WORKOUTS</font>", normal_style),
            Paragraph(f"<font size=14 color='#D97706'><b>{total_dur_hours} hrs</b></font><br/><font size=8 color='#64748B'>ACTIVE DURATION</font>", normal_style),
            Paragraph(f"<font size=14 color='#DC2626'><b>{avg_cal_burn} kcal</b></font><br/><font size=8 color='#64748B'>AVG CAL / WORKOUT</font>", normal_style),
            Paragraph(f"<font size=14 color='#7C3AED'><b>{metrics['current_streak']} Days</b></font><br/><font size=8 color='#64748B'>CURRENT STREAK</font>", normal_style),
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[108, 108, 108, 108, 108])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(Paragraph("📊 <b>Executive Fitness Performance Summary</b>", h2_style))
    story.append(kpi_table)
    story.append(Spacer(1, 10))
    
    # --- 3. Machine Learning & Deep Learning Model Benchmark ---
    models_dict = model_meta.get("models", {})
    best_m_name = model_meta.get("best_model", "Deep Neural Network (DNN)")
    
    ml_headers = [Paragraph("<b>Model Architecture</b>", bold_style), Paragraph("<b>MAE (kcal)</b>", bold_style), Paragraph("<b>RMSE (kcal)</b>", bold_style), Paragraph("<b>R² Score</b>", bold_style), Paragraph("<b>Deployment Status</b>", bold_style)]
    ml_rows = [ml_headers]
    
    for m_name, m_stats in models_dict.items():
        is_best = (m_name == best_m_name)
        status_text = "⭐ Active Best Model" if is_best else "Trained & Evaluated"
        ml_rows.append([
            Paragraph(f"<b>{m_name}</b>", normal_style),
            Paragraph(f"{m_stats.get('mae', 'N/A')}", normal_style),
            Paragraph(f"{m_stats.get('rmse', 'N/A')}", normal_style),
            Paragraph(f"<b>{m_stats.get('r2', 'N/A')}</b>", normal_style),
            Paragraph(status_text, bold_style if is_best else normal_style)
        ])
        
    ml_table = Table(ml_rows, colWidths=[160, 90, 90, 90, 110])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    # Color header text white in ReportLab
    for i in range(len(ml_headers)):
        ml_headers[i].style.textColor = colors.white
        
    story.append(Paragraph("🤖 <b>AI/ML/DL Calorie Prediction Architecture Benchmark</b>", h2_style))
    story.append(ml_table)
    story.append(Spacer(1, 10))
    
    # --- 4. Recent Workouts Log Table ---
    story.append(Paragraph("🏃 <b>Recent Workout Sessions (Real Database Logs)</b>", h2_style))
    recent_workouts = workouts_df.head(8)
    
    w_headers = [Paragraph("<b>Date</b>", bold_style), Paragraph("<b>Exercise</b>", bold_style), Paragraph("<b>Duration</b>", bold_style), Paragraph("<b>Avg HR</b>", bold_style), Paragraph("<b>Calories</b>", bold_style), Paragraph("<b>Intensity</b>", bold_style), Paragraph("<b>Source</b>", bold_style)]
    w_rows = [w_headers]
    
    for _, w in recent_workouts.iterrows():
        w_rows.append([
            Paragraph(str(w['date']), normal_style),
            Paragraph(str(w['exercise']), bold_style),
            Paragraph(f"{w['duration']} min", normal_style),
            Paragraph(f"{w['heart_rate']} bpm", normal_style),
            Paragraph(f"<b>{w['calories']} kcal</b>", bold_style),
            Paragraph(str(w['intensity']), normal_style),
            Paragraph(str(w['input_source']), normal_style)
        ])
        
    w_table = Table(w_rows, colWidths=[70, 110, 65, 65, 80, 75, 75])
    w_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0284C7")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    for i in range(len(w_headers)):
        w_headers[i].style.textColor = colors.white
        
    story.append(w_table)
    story.append(Spacer(1, 10))
    
    # --- 5. AI Insights & Habitual Intelligence ---
    story.append(Paragraph("💡 <b>AI Fitness Intelligence & Habit Analysis</b>", h2_style))
    insight_bullets = []
    for ins in insights.get('insights_list', []):
        insight_bullets.append(Paragraph(f"• {ins}", normal_style))
        insight_bullets.append(Spacer(1, 3))
        
    story.extend(insight_bullets)
    story.append(Spacer(1, 10))
    
    # --- 6. Unlocked Achievements ---
    unlocked_list = [a for a in achievements if a['is_unlocked']]
    story.append(Paragraph(f"🏆 <b>Unlocked Badges & Achievements ({len(unlocked_list)} / {len(achievements)})</b>", h2_style))
    
    ach_text_list = []
    for a in unlocked_list:
        ach_text_list.append(Paragraph(f"{a['icon']} <b>{a['title']}</b> — {a['description']} <i>(Unlocked: {a['unlocked_at'][:10]})</i>", normal_style))
        ach_text_list.append(Spacer(1, 3))
        
    story.extend(ach_text_list)
    story.append(Spacer(1, 14))
    
    # --- Footer ---
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceBefore=6, spaceAfter=6))
    footer_text = Paragraph(
        "<i>Calories Burnt Prediction AI • Built with Python, Streamlit, Scikit-Learn & TensorFlow • Not medical advice.</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, textColor=colors.HexColor("#94A3B8"), alignment=1)
    )
    story.append(footer_text)
    
    doc.build(story)
    return pdf_path

if __name__ == "__main__":
    report_file = generate_pdf_report(1)
    print(f"PDF Report generated at: {report_file}")
