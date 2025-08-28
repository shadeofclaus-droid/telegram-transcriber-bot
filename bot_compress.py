import asyncio
import os
import tempfile
import subprocess
import logging
from pathlib import Path
from contextlib import suppress

from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from openai import OpenAI

# ---------- config ----------
MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB allowed for incoming media
TARGET_SR = 16000
MAX_API_FILE_BYTES = 25 * 1024 * 1024  # 25 MB limit for OpenAI transcription
# ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("transcriber")

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var.")

client = OpenAI(api_key=OPENAI_API_KEY)

def run_ffmpeg_to_mp3(src: Path, dst: Path, sr: int = TARGET_SR, bitrate: str = "64k"):
    """Convert an audio/video file to mono MP3 with a given sample rate and bitrate."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-b:a", bitrate,
        str(dst)
    ]
    log.info("FFmpeg compress: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def human_size(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.1f}{units[i]}"

def transcribe_audio(path: Path, language: str | None = None) -> str:
    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language
        )
    return r.text or ""

async def start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Надішли voice / audio / відео або файл з аудіо — поверну текст.\n"
        "Можна додати підпис: lang=uk або lang=en."
    )

async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — привітання\n"
        "/help — як користуватися\n"
        "/ping — перевірка доступності\n"
        "/privacy — політика приватності (коротко)\n\n"
        "Надішли аудіо/voice/відео/файл — отримаєш розшифровку.\n"
        "Порада: додай підпис lang=uk або lang=en."
    )

async def ping(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")

async def privacy(update: Update, _: ContextTypes.DEFAULT_TYPE):
    txt = (
        "<b>Приватність</b>\n"
        "Файли зберігаються тимчасово лише для конвертації й видаляються одразу після транскрипції. "
        "Текст відповіді надсилається в цей чат і ніде більше не публікується."
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)

AUDIO_FILTER = (
    filters.VOICE | filters.AUDIO | filters.VIDEO |
    filters.Document.AUDIO |
    filters.Document.MimeType("audio/") | filters.Document.MimeType("video/") |
    filters.Document.FileExtension("m4a") | filters.Document.FileExtension("mp3") |
    filters.Document.FileExtension("wav") | filters.Document.FileExtension("ogg") |
    filters.Document.FileExtension("oga") | filters.Document.FileExtension("opus") |
    filters.Document.FileExtension("flac")| filters.Document.FileExtension("aac") |
    filters.Document.FileExtension("wma") | filters.Document.FileExtension("mkv") |
    filters.Document.FileExtension("mp4") | filters.Document.FileExtension("mov")
)

async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    with suppress(Exception):
        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    file = None
    filename = "input"
    try:
        if msg.voice:
            file = await msg.voice.get_file()
            filename = "voice.ogg"
        elif msg.audio:
            file = await msg.audio.get_file()
            filename = msg.audio.file_name or "audio.bin"
        elif msg.video:
            file = await msg.video.get_file()
            filename = msg.video.file_name or "video.mp4"
        elif msg.document:
            file = await msg.document.get_file()
            filename = msg.document.file_name or "file.bin"
        else:
            await msg.reply_text("Надішліть аудіо/voice/відео/файл.")
            return
    except Exception as e:
        log.exception("get_file failed")
        await msg.reply_text(f"Не вдалося отримати файл: {e}")
        return

    language = None
    if msg.caption:
        for p in msg.caption.split():
            if p.lower().startswith("lang="):
                language = p.split("=",1)[1].strip() or None

    try:
        fsize = getattr(file, "file_size", None)
        if fsize and fsize > MAX_FILE_BYTES:
            await msg.reply_text(f"Файл завеликий ({human_size(fsize)}). Обмеження {human_size(MAX_FILE_BYTES)}.")
            return
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as td_name:
        td = Path(td_name)
        src = td / filename
        compressed = td / "converted.mp3"

        try:
            await file.download_to_drive(custom_path=str(src))
            log.info("Downloaded to %s", src)
        except Exception as e:
            log.exception("download_to_drive failed")
            await msg.reply_text(f"Помилка завантаження файлу: {e}")
            return

        # Compress to mp3
        try:
            run_ffmpeg_to_mp3(src, compressed, sr=TARGET_SR, bitrate="64k")
        except subprocess.CalledProcessError:
            await msg.reply_text("Не вдалося обробити аудіо через ffmpeg.")
            return
        except FileNotFoundError:
            await msg.reply_text("ffmpeg не знайдено на сервері. Перевір конфігурацію деплою.")
            return

        # If bigger than allowed, try lower bitrate
        try:
            if compressed.stat().st_size > MAX_API_FILE_BYTES:
                log.info("compressed size %s exceeds limit; re-encoding with 32k bitrate", compressed.stat().st_size)
                run_ffmpeg_to_mp3(src, compressed, sr=TARGET_SR, bitrate="32k")
        except Exception as e:
            log.exception("re-encode failed")
            await msg.reply_text(f"Помилка стиснення файлу: {e}")
            return

        if compressed.stat().st_size > MAX_API_FILE_BYTES:
            await msg.reply_text(
                "Аудіо занадто довге або не вдається стиснути до допустимого розміру (25 MB). "
                "Спробуйте надіслати коротший або більш стислий файл."
            )
            return

        try:
            text = await asyncio.to_thread(transcribe_audio, compressed, language)
        except Exception as e:
            log.exception("transcribe failed")
            await msg.reply_text(f"Помилка транскрипції: {e}")
            return

        text = (text or "").strip()
        if not text:
            await msg.reply_text("Не вдалось розпізнати мовлення (порожній результат).")
            return

        if len(text) > 3500:
            txt_path = td / "transcript.txt"
            txt_path.write_text(text, encoding="utf-8")
            await msg.reply_document(document=InputFile(str(txt_path), filename="transcript.txt"),
                                     caption="Розшифровка")
        else:
            await msg.reply_text(text)

def main():
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("privacy", privacy))
    app.add_handler(MessageHandler(AUDIO_FILTER, handle_media))
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
