# app/utils/mail_util.py

import smtplib
from email.mime.text import MIMEText
from app.config import GMAIL_SMTP_CONFIG

def send_email(receiver_email, code):
    sender_email = GMAIL_SMTP_CONFIG["sender_email"]
    sender_password = GMAIL_SMTP_CONFIG["sender_password"]

    subject = "驗證碼通知"
    body = f"您的驗證碼是：{code}，10 分鐘內有效"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
