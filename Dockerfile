FROM python:3.11-slim

# Неінтерактивний apt
ENV DEBIAN_FRONTEND=noninteractive

# Встановити системні залежності + ffmpeg
RUN apt-get update  && apt-get install -y --no-install-recommends ffmpeg ca-certificates  && rm -rf /var/lib/apt/lists/*

# Робоча директорія
WORKDIR /app

# Кеш установки Python залежностей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код бота
COPY bot.py .

COPY bot_improved.py .
COPY bot_compress.py . 
COPY bot_dynamic.py . 
COPY bot_dynamic_fix.py .
COPY .env .env

# Порт (не обов'язково для Telegram ботів, але інколи вимагає платформа)
EXPOSE 8080

# Встановити основну команду запуску бота
CMD python bot_improved.py
