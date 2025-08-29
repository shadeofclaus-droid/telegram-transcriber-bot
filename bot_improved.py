"""
Improved Telegram transcription bot with progressive audio compression,
inline settings, and post‑transcription actions such as summary, translation
and text export.  The bot responds to voice/audio/video/file messages,
compresses audio to stay within Whisper API limits, calls OpenAI
transcription, and returns either a chat message or a file with
interactive buttons for further actions.  User preferences for language
and output format are stored per chat via ``context.chat_data``.

Note: Summary and translation are implemented via OpenAI's Chat
Completions API.  Ensure your API key has sufficient quota.
"""

import asyncio
import os
import tempfile
import subprocess
import logging
import uuid
from io import BytesIO
from pathlib import Path
from contextlib import suppress
from typing import Optional, Dict

from dotenv import load_dotenv
from telegram import (
    Update,
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI


# ---------- configuration ----------
# Maximum original file size the bot will accept (in bytes).  Files larger
# than this will be rejected before attempting conversion.
MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB
# Maximum size of audio to send to the OpenAI API (in bytes).  The bot
# compresses audio progressively until it fits this limit.
MAX_API_FILE_BYTES = 25 * 1024 * 1024  # 25 MB
# Sample rate for conversion.
TARGET_SR = 16_000
# Bitrate ladder for progressive compression.  Lower values produce
# smaller files at the cost of quality.
BITRATE_LADDER = ["64k", "32k", "16k", "8k"]
# ---------------------------

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("transcriber_improved")

# Load environment variables for API keys
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def run_ffmpeg_to_mp3(src: Path, dst: Path, sr: int = TARGET_SR, bitrate: str = "64k") -> None:
    """Convert any audio/video file to mono MP3 at the given sample rate and bitrate.

    Uses ``ffmpeg`` command line.  Raises ``subprocess.CalledProcessError`` if
    conversion fails.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ac",
        "1",
        "-b:a",
        bitrate,
        "-ar",
        str(sr),
        str(dst),
    ]
    log.info("FFmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def human_size(n: int) -> str:
    """Return human readable file size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.1f}{units[i]}"


async def transcribe_audio(path_audio: Path, language: Optional[str] = None) -> str:
    """Call OpenAI Whisper API to transcribe the given audio file.

    Parameters
    ----------
    path_audio: Path
        Path to audio file (must be <= MAX_API_FILE_BYTES).
    language: str | None
        Optional language code (e.g. "uk", "en").  If None, auto‑detect.
    Returns
    -------
    str
        Transcribed text.
    """
    with open(path_audio, "rb") as f:
        # Whisper automatically detects language when language is None
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
        )
    return response.text or ""


async def generate_summary(text: str) -> str:
    """Generate a short summary of the given text using OpenAI Chat API.

    The summary is kept concise and uses 2–3 sentences.
    """
    # Guard against extremely long texts by truncating
    max_chars = 8000
    snippet = text[:max_chars]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that summarizes transcripts into a concise summary. "
                "Provide a short overview of the key points in 2–3 sentences."
            ),
        },
        {"role": "user", "content": snippet},
    ]
    try:
        # Use a smaller model to reduce latency and cost
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=256,
            temperature=0.5,
        )
        summary = response.choices[0].message.content.strip()
    except Exception as exc:
        log.exception("Summary generation failed: %s", exc)
        summary = "Не вдалося створити підсумок."
    return summary


async def translate_text(text: str, target_language: str) -> str:
    """Translate the given text to the target language using OpenAI Chat API.

    Supported target languages: "uk" (Ukrainian), "en" (English).
    """
    assert target_language in {"uk", "en"}
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. Translate the following text "
                f"to {'Ukrainian' if target_language == 'uk' else 'English'}. "
                "Keep names and numbers unchanged."
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
        )
        translation = response.choices[0].message.content.strip()
    except Exception as exc:
        log.exception("Translation failed: %s", exc)
        translation = "Не вдалося перекласти текст."
    return translation


def get_default_settings() -> Dict[str, str]:
    """Return default settings for a chat.

    language: "auto", "uk" or "en"
    output: "auto", "text" or "file"
    """
    return {"language": "auto", "output": "auto"}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command.

    Sends a welcome message with sample usage and quick access buttons.
    """
    chat_id = update.effective_chat.id
    # Initialize settings if not present
    settings = context.chat_data.setdefault("settings", get_default_settings())
    keyboard = [
        [
            InlineKeyboardButton("Налаштування", callback_data="settings_menu"),
            InlineKeyboardButton("Поради", callback_data="tips_menu"),
            InlineKeyboardButton("Політика", callback_data="privacy_menu"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = (
        "Привіт! Я транскрибую voice/audio/video у текст.\n"
        "Надішли файл або voice‑повідомлення.\n"
        "Можна додати підпис lang=uk або lang=en, або обрати мову в налаштуваннях."
    )
    await update.message.reply_text(text, reply_markup=reply_markup)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "/start — привітання\n"
        "/help — як користуватись\n"
        "/settings — налаштування (мова, формат виводу)\n"
        "/privacy — політика приватності\n\n"
        "Надішли аудіо/voice/відео або файл — отримаєш розшифровку.\n"
        "Порада: обери мову та формат у /settings."
    )


async def privacy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /privacy command."""
    txt = (
        "<b>Приватність</b>\n"
        "Файли зберігаються тимчасово лише для конвертації й видаляються одразу після транскрипції. "
        "Текст відповіді надсилається в цей чат і ніде більше не публікується."
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command by showing settings menu."""
    await show_settings_menu(update.effective_chat.id, context)


async def show_settings_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send an inline keyboard with current settings for language and output."""
    settings = context.chat_data.setdefault("settings", get_default_settings())
    lang = settings.get("language", "auto")
    output = settings.get("output", "auto")
    # Display current selections in button text
    lang_text = {
        "auto": "Мова: Авто",
        "uk": "Мова: Українська",
        "en": "Мова: English",
    }[lang]
    output_text = {
        "auto": "Формат: Авто",
        "text": "Формат: Текст",
        "file": "Формат: Файл",
    }[output]
    keyboard = [
        [InlineKeyboardButton(lang_text, callback_data="settings_lang"),
         InlineKeyboardButton(output_text, callback_data="settings_output")],
        [InlineKeyboardButton("⬅️ Закрити", callback_data="settings_close")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "Налаштування:\n"
            "• Мова визначає, яку мову Whisper має розпізнати.\n"
            "• Формат визначає, чи надсилати текст у чат, файл або автоматично."
        ),
        reply_markup=reply_markup,
    )


async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process callback queries originating from the settings menu."""
    query = update.callback_query
    if not query:
        return
    data = query.data
    settings = context.chat_data.setdefault("settings", get_default_settings())
    # Handle closing settings
    if data == "settings_close":
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=None)
        return
    # Toggle language menu
    if data == "settings_lang":
        # Show language options
        keyboard = [
            [InlineKeyboardButton("Авто", callback_data="set_lang_auto"),
             InlineKeyboardButton("Українська", callback_data="set_lang_uk"),
             InlineKeyboardButton("English", callback_data="set_lang_en")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="settings_back")],
        ]
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
        return
    # Toggle output menu
    if data == "settings_output":
        keyboard = [
            [InlineKeyboardButton("Авто", callback_data="set_output_auto"),
             InlineKeyboardButton("Текст", callback_data="set_output_text"),
             InlineKeyboardButton("Файл", callback_data="set_output_file")],
            [InlineKeyboardButton("⬅️ Назад", callback_data="settings_back")],
        ]
        await query.answer()
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
        return
    # Handle back navigation in settings
    if data == "settings_back":
        await query.answer()
        # Re‑render settings menu
        await query.delete_message()
        await show_settings_menu(update.effective_chat.id, context)
        return
    # Set language or output
    updated = False
    if data.startswith("set_lang_"):
        lang_val = data.split("set_lang_")[1]
        settings["language"] = lang_val
        updated = True
    elif data.startswith("set_output_"):
        output_val = data.split("set_output_")[1]
        settings["output"] = output_val
        updated = True
    if updated:
        context.chat_data["settings"] = settings
        await query.answer("Збережено ✅")
        # Return to settings overview
        await query.delete_message()
        await show_settings_menu(update.effective_chat.id, context)
    else:
        # Unhandled settings callback
        await query.answer()


async def handle_misc_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle other callback queries such as summary, translation, and downloads."""
    query = update.callback_query
    if not query or not query.data:
        return
    data = query.data
    await query.answer()
    # Data format: action:id[:lang]
    try:
        parts = data.split(":")
        action = parts[0]
        transcript_id = parts[1] if len(parts) > 1 else None
    except Exception:
        return
    transcripts: Dict[str, str] = context.chat_data.get("transcripts", {})
    text = transcripts.get(transcript_id)
    if not text:
        await query.message.reply_text("Вибачте, не знайдено текст.")
        return
    # Handle summary
    if action == "summary":
        summary = await generate_summary(text)
        await query.message.reply_text(summary)
        return
    # Handle translation
    if action == "translate":
        # Determine target language: if current setting is uk → en, else uk
        settings = context.chat_data.get("settings", get_default_settings())
        src_lang = settings.get("language", "auto")
        # If source language is auto, guess based on first few characters: assume if contains cyrillic then uk else en
        guessed = "uk" if any("а" <= c.lower() <= "я" for c in text[:200]) else "en"
        target = "en" if guessed == "uk" else "uk"
        translation = await translate_text(text, target)
        await query.message.reply_text(translation)
        return
    # Handle TXT download
    if action == "txt":
        bio = BytesIO(text.encode("utf-8"))
        bio.name = "transcript.txt"
        await query.message.reply_document(document=InputFile(bio), caption="transcript.txt")
        return
    # Fallback: ignore


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming media messages, compress audio and transcribe."""
    msg = update.message
    if not msg:
        return
    chat_id = msg.chat_id
    # Send immediate feedback to show processing has started
    with suppress(Exception):
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        status_msg = await msg.reply_text("Прийняв 🎧 Обробляю…", quote=True)
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
    # Read language from caption or settings
    language: Optional[str] = None
    # Check caption for lang=
    if msg.caption:
        for p in msg.caption.split():
            if p.lower().startswith("lang="):
                language = p.split("=", 1)[1].strip() or None
    # Override with settings if not specified
    settings = context.chat_data.setdefault("settings", get_default_settings())
    lang_pref = settings.get("language", "auto")
    if language is None and lang_pref != "auto":
        language = lang_pref
    # File size guard
    try:
        fsize = getattr(file, "file_size", None)
        if fsize and fsize > MAX_FILE_BYTES:
            await msg.reply_text(
                f"Файл завеликий ({human_size(fsize)}). Обмеження {human_size(MAX_FILE_BYTES)}."
            )
            return
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / filename
        compressed = td_path / "compressed.mp3"
        try:
            await file.download_to_drive(custom_path=str(src))
            log.info("Downloaded to %s", src)
        except Exception as e:
            log.exception("download_to_drive failed")
            await msg.reply_text(f"Помилка завантаження файлу: {e}")
            return
        # Progressive compression using bitrate ladder
        success = False
        for bitrate in BITRATE_LADDER:
            try:
                run_ffmpeg_to_mp3(src, compressed, sr=TARGET_SR, bitrate=bitrate)
            except subprocess.CalledProcessError:
                await msg.reply_text("Не вдалося обробити аудіо через ffmpeg.")
                return
            except FileNotFoundError:
                await msg.reply_text(
                    "ffmpeg не знайдено на сервері. Перевір конфігурацію деплою."
                )
                return
            if compressed.stat().st_size <= MAX_API_FILE_BYTES:
                success = True
                break
        if not success:
            await msg.reply_text(
                "Аудіо занадто довге або не вдається стиснути до допустимого розміру (25 МБ). "
                "Спробуйте надіслати коротший або більш стиснутий файл."
            )
            return
        # Transcribe
        try:
            text = await transcribe_audio(compressed, language)
        except Exception as e:
            log.exception("transcribe failed")
            await msg.reply_text(f"Помилка транскрипції: {e}")
            return
        text = (text or "").strip()
        if not text:
            await msg.reply_text(
                "Не вдалось розпізнати мовлення (порожній результат)."
            )
            return
        # Store transcript for later actions
        transcripts: Dict[str, str] = context.chat_data.setdefault(
            "transcripts", {}
        )
        transcript_id = uuid.uuid4().hex
        transcripts[transcript_id] = text
        context.chat_data["transcripts"] = transcripts
        # Determine output preference
        output_pref = settings.get("output", "auto")
        # Determine if file output is needed
        use_file = False
        if output_pref == "file":
            use_file = True
        elif output_pref == "text":
            use_file = False
        else:
            # auto: if text too long use file
            use_file = len(text) > 3500
        # Prepare buttons for post actions
        buttons = [
            [
                InlineKeyboardButton("Підсумок", callback_data=f"summary:{transcript_id}"),
                InlineKeyboardButton("Переклад", callback_data=f"translate:{transcript_id}"),
            ],
            [InlineKeyboardButton("TXT", callback_data=f"txt:{transcript_id}")],
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        if use_file:
            bio = BytesIO(text.encode("utf-8"))
            bio.name = "transcript.txt"
            caption = "Розшифровка"
            await msg.reply_document(document=InputFile(bio), caption=caption, reply_markup=reply_markup)
        else:
            await msg.reply_text(text, reply_markup=reply_markup)
        # Delete status message if it exists
        with suppress(Exception):
            await status_msg.delete()


async def tips_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send some helpful tips when user presses the tips button."""
    query = update.callback_query
    if query:
        await query.answer()
        await query.message.reply_text(
            "Поради:\n"
            "• Додавай lang=uk або lang=en у підписі до повідомлення.\n"
            "• У Налаштуваннях можна змінити мову та формат.\n"
            "• Щоб отримати підсумок або переклад, натисни відповідні кнопки після розшифровки."
        )


async def button_dispatcher(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch callback queries to appropriate handlers based on prefix."""
    data = update.callback_query.data if update.callback_query else ""
    if data.startswith("settings") or data.startswith("set_"):
        await handle_settings_callback(update, context)
    elif data.startswith("summary") or data.startswith("translate") or data.startswith("txt"):
        await handle_misc_callback(update, context)
    elif data == "tips_menu":
        await tips_callback(update, context)
    elif data == "privacy_menu":
        await privacy(update.callback_query, context)
    elif data == "settings_menu":
        await settings_cmd(update.callback_query, context)
    else:
        await update.callback_query.answer()


def main() -> None:
    """Start the Telegram bot application."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CommandHandler("privacy", privacy))
    # Message handler for media
    AUDIO_FILTER = (
        filters.VOICE
        | filters.AUDIO
        | filters.VIDEO
        | filters.Document.AUDIO
        | filters.Document.MimeType("audio/")
        | filters.Document.MimeType("video/")
        | filters.Document.FileExtension("m4a")
        | filters.Document.FileExtension("mp3")
        | filters.Document.FileExtension("wav")
        | filters.Document.FileExtension("ogg")
        | filters.Document.FileExtension("oga")
        | filters.Document.FileExtension("opus")
        | filters.Document.FileExtension("flac")
        | filters.Document.FileExtension("aac")
        | filters.Document.FileExtension("wma")
        | filters.Document.FileExtension("mkv")
        | filters.Document.FileExtension("mp4")
        | filters.Document.FileExtension("mov")
    )
    app.add_handler(MessageHandler(AUDIO_FILTER, handle_media))
    # Callback query handler
    app.add_handler(CallbackQueryHandler(button_dispatcher))
    # Run the bot
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
