import os

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8384647509:AAG9yVwaQWm2fG8aje1RGTkOPnHalM0C_lE")
WINGO_API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@host:port/database")


