#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Telegram Transcriber Bot (2025-08-30)

This bot improves upon a basic transcription bot by adding:

* Progressive compression to keep audio under OpenAI's 25 MB upload limit.
* Support for long recordings via automatic chunking and stitching of timestamps.
* Word-level timestamps and generation of SRT subtitle files with a configurable
  number of words per caption.
* Interactive inline keyboard for downloading full text, SRT presets, custom
  caption lengths, or deleting the transcript.

Usage: set the environment variables `TELEGRAM_BOT_TOKEN` and `OPENAI_API_KEY`,
then run this script. Send voice, audio, video, or audio file attachments
to the bot; after transcription you can download the full text or subtitles.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from telegram import InputFile, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from openai import OpenAI

# -------- configuration --------
# Maximum size of incoming files accepted by the bot (Telegram limit)
MAX_INCOMING_BYTES = 200 * 1024 * 1024  # 200 MB
# Maximum size of audio uploaded to OpenAI API
MAX_API_BYTES = 25 * 1024 * 1024  # 25 MB
# Target sample rate when converting audio
TARGET_SR = 16000
# Threshold in seconds beyond which we split the recording into chunks
LONG_RECORDING_SEC = 20 * 60  # 20 minutes
# Duration of each chunk when splitting long recordings
CHUNK_SEC = 15 * 60  # 15 minutes
# Default number of words per caption in SRT files
DEFAULT_WPC = 8
# --------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("improved_bot")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var.")

# Instantiate OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

def human_size(n: int) -> str:
    """Return a human‑readable file size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    i = 0
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.1f}{units[i]}"


def run_ffmpeg_to_wav(src: Path, dst: Path, sr: int = TARGET_SR) -> None:
    """Convert media file to mono WAV with given sample rate using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        str(dst),
    ]
    log.info("FFmpeg to WAV: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_ffmpeg_to_mp3(src: Path, dst: Path, sr: int = TARGET_SR, bitrate: str = "64k") -> None:
    """Convert audio to mono MP3 with specified bitrate."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-b:a",
        bitrate,
        str(dst),
    ]
    log.info("FFmpeg to MP3: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compress_audio_under_limit(src: Path) -> Path:
    """
    Compress a WAV file to MP3, progressively reducing bitrate until
    the resulting file is within the API upload limit. Returns the path
    to the compressed file.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        compressed = Path(tmp.name)
    bitrates = ["64k", "48k", "32k", "16k", "8k"]
    for br in bitrates:
        try:
            run_ffmpeg_to_mp3(src, compressed, sr=TARGET_SR, bitrate=br)
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in container.")
        except subprocess.CalledProcessError:
            continue
        if compressed.stat().st_size <= MAX_API_BYTES:
            return compressed
    # If no bitrate fits, return the last compressed file anyway
    return compressed


def media_duration_seconds(path: Path) -> float:
    """Return duration of media file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def split_wav_to_chunks(src_wav: Path, out_dir: Path, chunk_sec: int = CHUNK_SEC) -> List[Path]:
    """
    Split a WAV file into segments of roughly chunk_sec seconds using ffmpeg.
    Returns a sorted list of chunk file paths.
    """
    pattern = out_dir / "chunk_%03d.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_wav),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-c",
        "copy",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(out_dir.glob("chunk_*.wav"))


def transcribe_verbose(path: Path, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload audio to Whisper and return the verbose JSON response with
    word- and segment-level timestamps.
    """
    with open(path, "rb") as f:
        log.info(
            "Uploading audio to OpenAI: %s (%.2f MB)",
            path.name,
            path.stat().st_size / (1024 * 1024),
        )
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="verbose_json",
        )
    # The OpenAI SDK returns a dict-like object or a dataclass; convert to dict
    if isinstance(response, dict):
        return response
    try:
        return response.to_dict()
    except Exception:
        return json.loads(str(response))


def transcribe_many_chunks(
    chunks: List[Path], language: Optional[str], status_edit
) -> Dict[str, Any]:
    """
    Transcribe multiple audio chunks sequentially. The callback `status_edit`
    should be a function that accepts a string to update the status message.
    All word and segment timestamps are shifted by the cumulative duration of
    preceding chunks.
    """
    total_text: List[str] = []
    all_words: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    offset = 0.0
    for i, ch in enumerate(chunks, 1):
        status_edit(f"🧠 Транскрибую… (частина {i}/{len(chunks)})")
        candidate = compress_audio_under_limit(ch)
        result = transcribe_verbose(candidate, language)
        text = (result.get("text") or "").strip()
        if text:
            total_text.append(text)
        # shift words
        for w in result.get("words") or []:
            all_words.append({"start": w["start"] + offset, "end": w["end"] + offset, "word": w["word"]})
        # shift segments
        for s in result.get("segments") or []:
            all_segments.append({"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"]})
        # update offset using original chunk duration
        dur = media_duration_seconds(ch)
        offset += dur if dur > 0 else 0
    return {
        "text": "\n".join(total_text).strip(),
        "words": all_words,
        "segments": all_segments,
    }


def _fmt_ts(ts: float) -> str:
    """Format a timestamp in seconds to SRT HH:MM:SS,mmm."""
    hours, rem = divmod(int(ts), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int(round((ts - int(ts)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def make_srt_from_words(words: List[Dict[str, Any]], words_per_caption: int = DEFAULT_WPC) -> str:
    """
    Create an SRT subtitle string from a list of word entries. Each entry in
    `words` should have keys: ``start``, ``end`` and ``word``.
    Words are grouped into captions of up to `words_per_caption` words.
    """
    cues = []
    chunk: List[Dict[str, Any]] = []
    for w in words:
        # attach punctuation to previous token
        if w["word"] in {".", ",", "!", "?", ":", ";", "…"} and chunk:
            chunk[-1]["word"] += w["word"]
            chunk[-1]["end"] = w["end"]
        else:
            chunk.append(dict(w))
        if len(chunk) >= words_per_caption:
            cues.append(chunk)
            chunk = []
    if chunk:
        cues.append(chunk)
    lines: List[str] = []
    for i, c in enumerate(cues, 1):
        start = c[0]["start"]
        end = c[-1]["end"]
        text = " ".join(tok["word"] for tok in c)
        lines.append(str(i))
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(text)
        lines.append("")  # blank line between cues
    return "\n".join(lines)


def build_actions_kb(chat_id: int, wpc_default: int = DEFAULT_WPC) -> InlineKeyboardMarkup:
    """
    Construct an inline keyboard for choosing actions after transcription.
    ``chat_id`` serves as a unique identifier for the stored transcript.
    """
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📝 Текст", callback_data=f"text:{chat_id}")],
        [
            InlineKeyboardButton("🎬 Субтитри (4)", callback_data=f"srt:4:{chat_id}"),
            InlineKeyboardButton("🎬 Субтитри (6)", callback_data=f"srt:6:{chat_id}"),
            InlineKeyboardButton("🎬 Субтитри (8)", callback_data=f"srt:8:{chat_id}"),
        ],
        [
            InlineKeyboardButton("🎬 Субтитри (10)", callback_data=f"srt:10:{chat_id}"),
            InlineKeyboardButton("🎬 Субтитри (12)", callback_data=f"srt:12:{chat_id}"),
        ],
        [
            InlineKeyboardButton("🔢 Ввести число…", callback_data=f"custom:{chat_id}"),
            InlineKeyboardButton("🗑️ Видалити", callback_data=f"delete:{chat_id}"),
        ],
    ])


# Command handlers
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Надішли voice/audio/відео або файл з аудіо — і я поверну текст. "
        "Можна додати підпис: lang=uk або lang=en.\n"
        "Після транскрипції ти зможеш завантажити субтитри (SRT)."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — привітання\n"
        "/help — як користуватися\n"
        "/ping — перевірка доступності\n"
        "/privacy — політика приватності\n\n"
        "Надішли аудіо/voice/відео/файл — і отримаєш розшифровку.\n"
        "Після транскрипції можна завантажити SRT з різною кількістю слів у таймкоді."
    )


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong ✅")


async def privacy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "<b>Приватність</b>\n"
        "Файли зберігаються тимчасово лише для конвертації й видаляються одразу після транскрипції. "
        "Текст відповіді надсилається тільки в цей чат і ніде більше не публікується."
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.HTML)


# Filter for media messages
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


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    # Show typing indicator
    with suppress(Exception):
        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    # Determine file and filename
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
        await msg.reply_text("Надішліть аудіо/voice/відео/файл.")
        return

    # Parse language from caption (optional parameter like lang=uk)
    language: Optional[str] = None
    if msg.caption:
        for part in msg.caption.split():
            if part.lower().startswith("lang="):
                language = part.split("=", 1)[1].strip() or None

    # Check file size
    try:
        fsize = getattr(file, "file_size", None)
        if fsize and fsize > MAX_INCOMING_BYTES:
            await msg.reply_text(
                f"Файл завеликий ({human_size(fsize)}). Обмеження {human_size(MAX_INCOMING_BYTES)}."
            )
            return
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / filename
        wav = td_path / "converted.wav"

        # Download file
        try:
            await file.download_to_drive(custom_path=str(src))
            log.info("Downloaded to %s", src)
        except Exception as e:
            log.exception("download_to_drive failed")
            await msg.reply_text(f"Помилка завантаження файлу: {e}")
            return

        # Convert to WAV
        try:
            run_ffmpeg_to_wav(src, wav, sr=TARGET_SR)
        except FileNotFoundError:
            await msg.reply_text(
                "ffmpeg не знайдено на сервері. Перевір конфігурацію деплою."
            )
            return
        except subprocess.CalledProcessError:
            await msg.reply_text("Не вдалося обробити аудіо через ffmpeg.")
            return

        # Compute duration to decide whether to split
        duration_sec = media_duration_seconds(wav)

        # Send status message
        status = await msg.reply_text("🗜️ Стискаю до ліміту…")

        try:
            if duration_sec <= LONG_RECORDING_SEC:
                # Short recording: compress once and transcribe
                candidate = compress_audio_under_limit(wav)
                await status.edit_text("🧠 Транскрибую…")
                result = await asyncio.to_thread(transcribe_verbose, candidate, language)
            else:
                # Long recording: split into chunks and transcribe each
                await status.edit_text("✂️ Довгий файл — ділю на частини…")
                chunks = split_wav_to_chunks(wav, td_path, chunk_sec=CHUNK_SEC)

                def status_update(text: str) -> None:
                    # Use thread-safe coroutine scheduling to update the status message
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        return
                    coro = status.edit_text(text)
                    asyncio.run_coroutine_threadsafe(coro, loop)

                result = await asyncio.to_thread(
                    transcribe_many_chunks, chunks, language, status_update
                )
        except Exception as e:
            log.exception("transcription failed")
            await status.edit_text(f"Помилка транскрипції: {e}")
            return

        # Store transcript in chat_data using a timestamp as a unique key
        tid = int(time.time() * 1000)
        context.chat_data.setdefault("transcript_text", {})[tid] = result.get("text") or ""
        context.chat_data.setdefault("transcript_words", {})[tid] = result.get("words") or []

        # Send transcription (text or file)
        text = (result.get("text") or "").strip()
        if not text:
            await status.edit_text(
                "Не вдалось розпізнати мовлення (порожній результат)."
            )
        else:
            if len(text) > 3500:
                txt_path = td_path / "transcript.txt"
                txt_path.write_text(text, encoding="utf-8")
                await context.bot.send_chat_action(
                    chat_id=msg.chat_id, action=ChatAction.UPLOAD_DOCUMENT
                )
                await msg.reply_document(
                    document=InputFile(str(txt_path), filename="transcript.txt"),
                    caption="Розшифровка",
                )
            else:
                await status.edit_text(text)

        # Send action menu
        kb = build_actions_kb(tid)
        await msg.reply_text("Обери наступну дію ↓", reply_markup=kb)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button presses."""
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    parts = data.split(":")
    if not parts:
        return
    action = parts[0]
    if action == "text":
        # Send full text transcript
        if len(parts) < 2:
            return
        tid = int(parts[1])
        text = context.chat_data.get("transcript_text", {}).get(tid)
        if text:
            if len(text) > 4096:
                # Send as file when too long
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                    tmp.write(text.encode("utf-8"))
                    tmp_path = Path(tmp.name)
                await query.message.reply_document(
                    document=InputFile(str(tmp_path), filename="transcript.txt"),
                    caption="Розшифровка",
                )
            else:
                await query.message.reply_text(text)
        else:
            await query.message.reply_text("Немає тексту для цього запису.")
    elif action == "srt":
        # Generate SRT with given number of words per caption
        if len(parts) < 3:
            return
        try:
            n = int(parts[1])
            tid = int(parts[2])
        except Exception:
            return
        words = context.chat_data.get("transcript_words", {}).get(tid, [])
        if not words:
            await query.message.reply_text("Немає таймкодів для цього запису.")
            return
        srt = make_srt_from_words(words, n)
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
            tmp.write(srt.encode("utf-8"))
            tmp_path = Path(tmp.name)
        await context.bot.send_chat_action(
            chat_id=query.message.chat_id, action=ChatAction.UPLOAD_DOCUMENT
        )
        await query.message.reply_document(
            document=InputFile(str(tmp_path), filename=f"subtitles_{n}w.srt"),
            caption=f"Субтитри ({n} слів/таймкод)",
        )
        # Remember last chosen WPC
        context.chat_data["wpc_default"] = n
    elif action == "custom":
        # Prompt user to enter a custom number of words
        if len(parts) < 2:
            return
        tid = int(parts[1])
        context.chat_data["awaiting_wpc_for"] = tid
        await query.message.reply_text("Введи кількість слів на таймкод (1-30):")
    elif action == "delete":
        # Delete stored transcript
        if len(parts) < 2:
            return
        tid = int(parts[1])
        context.chat_data.get("transcript_text", {}).pop(tid, None)
        context.chat_data.get("transcript_words", {}).pop(tid, None)
        await query.message.reply_text("Транскрипт видалено.")


async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages. Used for entering custom words per caption."""
    msg = update.message
    if not msg:
        return
    tid = context.chat_data.get("awaiting_wpc_for")
    if not tid:
        return
    try:
        n = int(msg.text.strip())
        if n < 1 or n > 30:
            raise ValueError
    except Exception:
        await msg.reply_text("Будь ласка, введи число від 1 до 30.")
        return
    # Generate SRT
    words = context.chat_data.get("transcript_words", {}).get(tid, [])
    if not words:
        await msg.reply_text("Немає таймкодів для цього запису.")
        context.chat_data.pop("awaiting_wpc_for", None)
        return
    srt = make_srt_from_words(words, n)
    from io import BytesIO
    bio = BytesIO(srt.encode("utf-8"))
    bio.name = f"subtitles_{n}w.srt"
    await context.bot.send_chat_action(
        chat_id=msg.chat_id, action=ChatAction.UPLOAD_DOCUMENT
    )
    await msg.reply_document(
        document=InputFile(bio), caption=f"Субтитри ({n} слів/таймкод)"
    )
    # Remember last chosen value and clear awaiting flag
    context.chat_data["wpc_default"] = n
    context.chat_data.pop("awaiting_wpc_for", None)
    # Show menu again
    kb = build_actions_kb(tid, wpc_default=n)
    await msg.reply_text("Готово. Обери наступну дію ↓", reply_markup=kb)


async def main() -> None:
    """Entry point for the bot."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    # Register handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("privacy", privacy_cmd))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text))
    app.add_handler(MessageHandler(AUDIO_FILTER, handle_media))
    # Run the bot
    await app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    asyncio.run(main())