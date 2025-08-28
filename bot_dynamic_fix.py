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
MAX_API_FILE_BYTES = 25 * 1024 * 1024  # 25 MB limit for OpenAI API
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

def run_ffmpeg_to_mp3(src: Path, dst: Path, sr: int = TARGET_SR, bitrate: str = "64k") -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src),
        "-vn",
        "-ar", str(sr),
        "-ac", "1",
        "-b:a", bitrate,
        str(dst)
    ]
    log.info("FFmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    i = 0
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
        "\u041f\u0440\u0438\u0432\u0456\u0442! \u041d\u0430\u0434\u0456\u0448\u043b\u0438 voice / audio / \u0432\u0456\u0434\u0435\u043e \u0430\u0431\u043e \u0444\u0430\u0439\u043b \u0437 \u0430\u0443\u0434\u0456\u043e \u2014 \u043f\u043e\u0432\u0435\u0440\u043d\u0443 \u0442\u0435\u043a\u0441\u0442.\n"
        "\u041c\u043e\u0436\u043d\u0430 \u0434\u043e\u0434\u0430\u0442\u0438 \u043f\u0456\u0434\u043f\u0438\u0441: lang=uk \u0430\u0431\u043e lang=en."
    )

async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start \u2014 \u043f\u0440\u0438\u0432\u0456\u0442\u0430\u043d\u043d\u044f\n"
        "/help \u2014 \u044f\u043a \u043a\u043e\u0440\u0438\u0441\u0442\u0443\u0432\u0430\u0442\u0438\u0441\u044f\n"
        "/ping \u2014 \u043f\u0435\u0440\u0435\u0432\u0456\u0440\u043a\u0430 \u0434\u043e\u0441\u0442\u0443\u043f\u043d\u043e\u0441\u0442\u0456\n"
        "/privacy \u2014 \u043f\u043e\u043b\u0456\u0442\u0438\u043a\u0430 \u043f\u0440\u0438\u0432\u0430\u0442\u043d\u043e\u0441\u0442\u0456 (\u043a\u043e\u0440\u043e\u0442\u043a\u043e)\n\n"
        "\u041d\u0430\u0434\u0456\u0448\u043b\u0438 \u0430\u0443\u0434\u0456\u043e/voice/\u0432\u0456\u0434\u0435\u043e/\u0444\u0430\u0439\u043b \u2014 \u043e\u0442\u0440\u0438\u043c\u0430\u0454\u0448 \u0440\u043e\u0437\u0448\u0438\u0444\u0440\u043e\u0432\u043a\u0443.\n"
        "\u041f\u043e\u0440\u0430\u0434\u0430: \u0434\u043e\u0434\u0430\u0439 \u043f\u0456\u0434\u043f\u0438\u0441 lang=uk \u0430\u0431\u043e lang=en."
    )

async def ping(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong \u2705")

async def privacy(update: Update, _: ContextTypes.DEFAULT_TYPE):
    txt = (
        "<b>\u041f\u0440\u0438\u0432\u0430\u0442\u043d\u0456\u0441\u0442\u044c</b>\n"
        "\u0424\u0430\u0439\u043b\u0438 \u0437\u0431\u0435\u0440\u0456\u0433\u0430\u044e\u0442\u044c\u0441\u044f \u0442\u0438\u043c\u0447\u0430\u0441\u043e\u0432\u043e \u043b\u0438\u0448\u0435 \u0434\u043b\u044f \u043a\u043e\u043d\u0432\u0435\u0440\u0442\u0430\u0446\u0456\u0457 \u0439 \u0432\u0438\u0434\u0430\u043b\u044f\u044e\u0442\u044c\u0441\u044f \u043e\u0434\u0440\u0430\u0437\u0443 \u043f\u0456\u0441\u043b\u044f \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0446\u0456\u0457. "
        "\u0422\u0435\u043a\u0441\u0442 \u0432\u0456\u0434\u043f\u043e\u0432\u0456\u0434\u0456 \u043d\u0430\u0434\u0441\u0438\u043b\u0430\u0454\u0442\u044c\u0441\u044f \u0432 \u0446\u0435\u0439 \u0447\u0430\u0442 \u0456 \u043d\u0456\u0434\u0435 \u0431\u0456\u043b\u044c\u0448\u0435 \u043d\u0435 \u043f\u0443\u0431\u043b\u0456\u043a\u0443\u0454\u0442\u044c\u0441\u044f."
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

    # show typing action
    with suppress(Exception):
        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    # determine file and filename
    file = None
    filename = "input"
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
        await msg.reply_text("\u041d\u0430\u0434\u0456\u0448\u043b\u0456\u0442\u044c \u0430\u0443\u0434\u0456\u043e/voice/\u0432\u0456\u0434\u0435\u043e/\u0444\u0430\u0439\u043b.")
        return

    # parse language from caption
    language = None
    if msg.caption:
        for p in msg.caption.split():
            if p.lower().startswith("lang="):
                language = p.split("=", 1)[1].strip() or None

    # check file size before conversion
    try:
        fsize = getattr(file, "file_size", None)
        if fsize and fsize > MAX_FILE_BYTES:
            await msg.reply_text(f"\u0424\u0430\u0439\u043b \u0437\u0430\u0432\u0435\u043b\u0438\u043a\u0438\u0439 ({human_size(fsize)}). \u041e\u0431\u043c\u0435\u0436\u0435\u043d\u043d\u044f {human_size(MAX_FILE_BYTES)}.")
            return
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / filename
        compressed = td_path / "converted.mp3"

        # download source file
        try:
            await file.download_to_drive(custom_path=str(src))
            log.info("Downloaded to %s", src)
        except Exception as e:
            log.exception("download_to_drive failed")
            await msg.reply_text(f"\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0437\u0430\u0432\u0430\u043d\u0442\u0430\u0436\u0435\u043d\u043d\u044f \u0444\u0430\u0439\u043b\u0443: {e}")
            return

        # compress progressively
        bitrates = ["64k", "32k", "16k", "8k"]
        success = False
        for br in bitrates:
            try:
                run_ffmpeg_to_mp3(src, compressed, sr=TARGET_SR, bitrate=br)
            except subprocess.CalledProcessError:
                await msg.reply_text("\u041d\u0435 \u0432\u0434\u0430\u043b\u043e\u0441\u044f \u043e\u0431\u0440\u043e\u0431\u0438\u0442\u0438 \u0430\u0443\u0434\u0456\u043e \u0447\u0435\u0440\u0435\u0437 ffmpeg.")
                return
            except FileNotFoundError:
                await msg.reply_text("ffmpeg \u043d\u0435 \u0437\u043d\u0430\u0439\u0434\u0435\u043d\u043e \u043d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440\u0456. \u041f\u0435\u0440\u0435\u0432\u0456\u0440 \u043a\u043e\u043d\u0444\u0456\u0433\u0443\u0440\u0430\u0446\u0456\u044e \u0434\u0435\u043f\u043b\u043e\u044e.")
                return

            if compressed.stat().st_size <= MAX_API_FILE_BYTES:
                success = True
                break

        if not success:
            await msg.reply_text(
                "\u0410\u0443\u0434\u0456\u043e \u0437\u0430\u043d\u0430\u0434\u0442\u043e \u0434\u043e\u0432\u0433\u0435 \u0430\u0431\u043e \u043d\u0435 \u0432\u0434\u0430\u0454\u0442\u044c\u0441\u044f \u0441\u0442\u0438\u0441\u043a\u043d\u0443\u0442\u0438 \u0434\u043e \u0434\u043e\u043f\u0443\u0441\u0442\u0438\u043c\u043e\u0433\u043e \u0440\u043e\u0437\u043c\u0456\u0440\u0443 (25 MB). \u0421\u043f\u0440\u043e\u0431\u0443\u0439\u0442\u0435 \u043d\u0430\u0434\u0456\u0441\u043b\u0430\u0442\u0438 \u043a\u043e\u0440\u043e\u0442\u0448\u0438\u0439 \u0430\u0431\u043e \u0431\u0456\u043b\u044c\u0448 \u0441\u0442\u0438\u0441\u043d\u0443\u0442\u0438\u0439 \u0444\u0430\u0439\u043b."
            )
            return

        # transcribe
        try:
            text = await asyncio.to_thread(transcribe_audio, compressed, language)
        except Exception as e:
            log.exception("transcribe failed")
            await msg.reply_text(f"\u041f\u043e\u043c\u0438\u043b\u043a\u0430 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0446\u0456\u0457: {e}")
            return

        text = (text or "").strip()
        if not text:
            await msg.reply_text("\u041d\u0435 \u0432\u0434\u0430\u043b\u043e\u0441\u044c \u0440\u043e\u0437\u043f\u0456\u0437\u043d\u0430\u0442\u0438 \u043c\u043e\u0432\u043b\u0435\u043d\u043d\u044f (\u043f\u043e\u0440\u043e\u0436\u043d\u0456\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442).")
            return

        if len(text) > 3500:
            from io import BytesIO
            bio = BytesIO(text.encode("utf-8"))
            bio.name = "transcript.txt"
            await msg.reply_document(document=InputFile(bio), caption="\u0420\u043e\u0437\u0448\u0438\u0444\u0440\u043e\u0432\u043a\u0430")
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
