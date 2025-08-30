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
COPY .env .env

# Порт (не обов'язково для Telegram ботів, але інколи вимагає платформа)
EXPOSE 8080

# Старт
CMD ["python", "bot.py"]

# Use compressed bot
COPY bot_compress.py .
CMD ["python", "bot_compress.py"]
# Use dynamic compression bot
COPY bot_dynamic.py .
CMD ["python", "bot_dynamic.py"]
# Override with dynamic uppercase
COPY bot_dynamic.py .
CMD ["python", "bot_dynamic.py"]
COPY bot_improved.py .
CMD ["python", "bot_improved.py"] 


# Override with dynamic fix
COPY bot_dynamic_fix.py .
CMD ["python", "bot_dynamic_fix.py"]
COPY bot_improved.py .
CMD ["python", "bot_improved.py"]
