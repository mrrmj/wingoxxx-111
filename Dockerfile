FROM python:3.11-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
ENV DATABASE_URL="postgresql://user:password@host:port/database"
ENV ENCRYPTION_KEY="your_fernet_key"

CMD ["python", "main.py"]


