# app/config.py
import os

from flask.cli import load_dotenv

MYSQL_CONFIG = {
    "host": "10.8.0.1",
    "user": "system",
    "password": "!QAZ2wsx#EDC",
    "database": "tcmha",
    "charset": "utf8mb4"
}

load_dotenv()  # 讀取 .env 檔案

GMAIL_SMTP_CONFIG = {
    "sender_email": os.getenv("GMAIL_SENDER_EMAIL"),
    "sender_password": os.getenv("GMAIL_SENDER_PASSWORD"),
}

