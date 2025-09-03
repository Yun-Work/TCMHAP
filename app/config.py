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

# load_dotenv()  # 讀取 .env 檔案

GMAIL_SMTP_CONFIG = {
    "sender_email": "tkuim2025@gmail.com",
    "sender_password": "vcuf rchb jraj wgst",
}

