"""
Modular Notification System (3-Layer Architecture)
Layer 1: In-App UI Notifications
Layer 2: Browser Web Notifications (Web API JS)
Layer 3: External Push & Email Notifications (Pushbullet / Webhook / SMTP)
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NOTIFICATION_API_KEY = os.getenv("NOTIFICATION_API_KEY", "")
NOTIFICATION_APP_ID = os.getenv("NOTIFICATION_APP_ID", "")
NOTIFICATION_WEBHOOK_URL = os.getenv("NOTIFICATION_WEBHOOK_URL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "fitness-ai@antigravity.app")

class NotificationService:
    """
    Modular notification dispatcher supporting multi-channel delivery.
    """
    
    @staticmethod
    def get_browser_notification_js(title, body, icon="🔥"):
        """
        Generates client-side JavaScript snippet to trigger native browser notification.
        """
        safe_title = json.dumps(title)
        safe_body = json.dumps(body)
        
        js_code = f"""
        <script>
        (function() {{
            if (!("Notification" in window)) {{
                console.log("This browser does not support desktop notifications");
                return;
            }}
            if (Notification.permission === "granted") {{
                new Notification({safe_title}, {{
                    body: {safe_body},
                    icon: "https://img.icons8.com/fluency/96/fire-element.png"
                }});
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function (permission) {{
                    if (permission === "granted") {{
                        new Notification({safe_title}, {{
                            body: {safe_body},
                            icon: "https://img.icons8.com/fluency/96/fire-element.png"
                        }});
                    }}
                }});
            }}
        }})();
        </script>
        """
        return js_code

    @staticmethod
    def send_email_notification(to_email, subject, body_text):
        """
        Layer 3: Sends email notification via SMTP if configured in .env.
        """
        if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD or not to_email:
            return {
                "success": False, 
                "message": "SMTP credentials or recipient email not configured in .env / Settings."
            }
            
        try:
            msg = MIMEMultipart()
            msg['From'] = SMTP_FROM
            msg['To'] = to_email
            msg['Subject'] = f"Calories Burnt AI: {subject}"
            msg.attach(MIMEText(body_text, 'plain'))
            
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to_email, msg.as_string())
            server.quit()
            return {"success": True, "message": f"Email successfully dispatched to {to_email}"}
        except Exception as e:
            return {"success": False, "message": f"Email dispatch failed: {str(e)}"}

    @staticmethod
    def send_push_notification(title, message, channel_token=None):
        """
        Layer 3: Sends push notification via webhook or external notification gateway (Pushbullet / Pushover / Webhook).
        """
        webhook_url = NOTIFICATION_WEBHOOK_URL or channel_token
        
        if not webhook_url:
            return {
                "success": False,
                "message": "Push webhook URL / token not configured. Set NOTIFICATION_WEBHOOK_URL in .env or Settings."
            }
            
        payload = {
            "title": f"🔥 Calorie AI: {title}",
            "body": message,
            "message": message,
            "app_id": NOTIFICATION_APP_ID
        }
        
        try:
            headers = {"Content-Type": "application/json"}
            if NOTIFICATION_API_KEY:
                headers["Access-Token"] = NOTIFICATION_API_KEY
                headers["Authorization"] = f"Bearer {NOTIFICATION_API_KEY}"
                
            resp = requests.post(webhook_url, json=payload, headers=headers, timeout=8)
            if resp.status_code in [200, 201, 202, 204]:
                return {"success": True, "message": "Push notification sent successfully."}
            else:
                return {"success": False, "message": f"Gateway responded with HTTP {resp.status_code}"}
        except Exception as e:
            return {"success": False, "message": f"Push notification delivery failed: {str(e)}"}

    @classmethod
    def dispatch_reminder(cls, title, message, channel_type="In-App", user_settings=None):
        """
        Routes the notification to requested layer.
        """
        results = {"channel": channel_type, "status": "delivered"}
        
        if channel_type == "Email" and user_settings:
            email_addr = user_settings.get("email_address", "")
            res = cls.send_email_notification(email_addr, title, message)
            results.update(res)
        elif channel_type == "Push" and user_settings:
            token = user_settings.get("push_token", "")
            res = cls.send_push_notification(title, message, token)
            results.update(res)
        else:
            # In-App / Browser
            results["success"] = True
            results["message"] = f"In-App / Browser notification triggered: {title}"
            
        return results

if __name__ == "__main__":
    test_res = NotificationService.dispatch_reminder("Workout Test", "Time to burn some calories!")
    print("Notification Service Test:", test_res)
