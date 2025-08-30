#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram Transcriber Bot — improved (full, 2025-08-30)
- Progressive audio conversion/compression to stay under OpenAI upload limit
- Verbose transcription with word/segment timestamps when available
- Summary modes: short / long / minutes (meeting protocol), language-aware
- Telegram-safe HTML output + safe chunking for long messages
- Subtitles (SRT) with selectable words-per-cue (1–30), remembers last choice
- Robust SRT fallback: words -> segments -> linear distribution by duration
- Visible statuses; settings; TXT export
- NEW: Proper SRT newlines; return menu after custom WPC; auto split long files and stitch back; progress by parts
"""

import asyncio
import os
import tempfile
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from openai import OpenAI

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var.")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Limits / constants
# -----------------------------------------------------------------------------
MAX_API_FILE_BYTES = 25 * 1024 * 1024  # OpenAI limit ~25MB
AUDIO_SR = 16000
TMP_WAV = "converted.wav"

# Telegram message safe length for HTML (leave headroom for tags)
MAX_TG_HTML = 3900

# Long file split config (seconds)
SPLIT_THRESHOLD_SEC = 20 * 60     # files longer than this will be split
SPLIT_CHUNK_SEC = 15 * 60         # target chunk length

# -----------------------------------------------------------------------------
# Simple settings model
# -----------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "summary_lang": "auto",       # auto / uk / en
    "summary_mode": "short",      # short / long / minutes
    "translate_target": "uk",     # reserved
    "output": "auto",             # reserved
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def human_size(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f}{units[i]}"

def pick_output_lang(transcript_lang: Optional[str], settings: Dict, fallback="uk") -> str:
    pref = (settings or {}).get("summary_lang", "auto")
    if pref != "auto":
        return pref
    if transcript_lang:
        t = transcript_lang.lower()
        if t.startswith("uk"):
            return "uk"
        if t.startswith("en"):
            return "en"
    return fallback

# -----------------------------------------------------------------------------
# FFmpeg / FFprobe helpers
# -----------------------------------------------------------------------------
def run_ffmpeg_to_wav(src: Path, dst: Path, sr=AUDIO_SR):
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(dst)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_ffmpeg_to_ogg(src: Path, dst: Path, bitrate_kbps: int = 32, sr=AUDIO_SR):
    # Opus in OGG container; very small while speech quality is fine at 16–32 kbps mono
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k",
        str(dst)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def media_duration_seconds(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", str(path)],
            text=True
        ).strip()
        return float(out)
    except Exception:
        return 0.0

def ensure_under_limit(src: Path) -> Path:
    """
    Try progressively smaller bitrates (Opus) to get under OpenAI 25MB upload limit.
    Returns path to OGG that is under the limit, or last attempt if still too big.
    """
    if src.stat().st_size <= MAX_API_FILE_BYTES:
        return src
    tmp = src
    for kb in [64, 48, 32, 24, 16, 12, 8]:
        ogg = src.with_suffix(".ogg")
        try:
            run_ffmpeg_to_ogg(tmp, ogg, bitrate_kbps=kb)
        except subprocess.CalledProcessError:
            continue
        if ogg.stat().st_size <= MAX_API_FILE_BYTES:
            return ogg
        tmp = ogg
    return tmp

def split_wav_to_chunks(src_wav: Path, out_dir: Path, chunk_sec: int = SPLIT_CHUNK_SEC) -> List[Path]:
    """Split WAV to ~chunk_sec segments."""
    pattern = out_dir / "chunk_%03d.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(src_wav),
        "-f", "segment", "-segment_time", str(chunk_sec),
        "-c", "copy", str(pattern)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(out_dir.glob("chunk_*.wav"))

# -----------------------------------------------------------------------------
# Transcription
# -----------------------------------------------------------------------------
def transcribe_verbose(path_audio: Path, language: Optional[str] = None) -> Dict:
    """
    Return {
      'text': str,
      'words': [{'start':s,'end':e,'word':w},...],
      'segments': [{'start':s,'end':e,'text':t}, ...],
      'lang': 'uk'|'en'|...
    }
    Tries gpt-4o-mini-transcribe with word/segment timestamps first,
    falls back to whisper-1 verbose_json.
    """
    # 1) Try 4o-mini-transcribe with timestamps
    try:
        with open(path_audio, "rb") as f:
            r = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language=language,
                timestamp_granularities=["word", "segment"],
            )
        words = []
        if getattr(r, "words", None):
            for w in r.words:
                words.append({"start": float(w.start), "end": float(w.end), "word": w.word})
        segments = []
        if getattr(r, "segments", None):
            for s in r.segments:
                segments.append({"start": float(s.start), "end": float(s.end), "text": s.text})
        lang = getattr(r, "language", None) or None
        return {"text": r.text, "words": words, "segments": segments, "lang": lang}
    except Exception:
        pass

    # 2) Fall back to whisper-1 verbose_json
    with open(path_audio, "rb") as f:
        r = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="verbose_json",
        )
    if isinstance(r, dict):
        text = r.get("text", "")
        segs = r.get("segments", []) or []
        lang = r.get("language") or None
    else:
        text = getattr(r, "text", "") or ""
        segs = getattr(r, "segments", []) or []
        lang = getattr(r, "language", None) or None

    # Approximate word timings from segments
    words = []
    for seg in segs:
        start = float(seg["start"])
        end = float(seg["end"])
        seg_text = (seg.get("text") or "").strip()
        tokens = seg_text.split()
        if not tokens:
            continue
        dur = max(0.001, end - start)
        per = dur / len(tokens)
        t = start
        for w in tokens:
            w_start = t
            w_end = min(end, t + per)
            words.append({"start": w_start, "end": w_end, "word": w})
            t = w_end
    segments = [{"start": float(s["start"]), "end": float(s["end"]), "text": s.get("text","")} for s in segs]
    return {"text": text, "words": words, "segments": segments, "lang": lang}

# -----------------------------------------------------------------------------
# Summarization helpers
# -----------------------------------------------------------------------------
def chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    chunks, cur, total = [], [], 0
    for para in text.split("\n"):
        if total + len(para) + 1 > max_chars and cur:
            chunks.append("\n".join(cur)); cur = []; total = 0
        cur.append(para); total += len(para) + 1
    if cur: chunks.append("\n".join(cur))
    return chunks

def sys_prompt_for(mode: str, target: str) -> str:
    lang_name = "Ukrainian" if target == "uk" else "English"
    base = (
        f"Return output strictly in Telegram HTML in {lang_name}. "
        "Use only <b>, <i>, <u>, <s>, <code>, <pre>, <blockquote>, <tg-spoiler>. "
        "NO Markdown, no other tags. Headings must be '<b>Heading</b>' on its own line. "
        "Use the bullet • (U+2022), one item per new line. "
        "Bold the word 'Дедлайн:' or 'Deadline:' when present. Keep it paste-safe for Telegram."
    )
    if mode == "short":
        return base + " Produce 4–6 concise bullets with the key facts only."
    if mode == "long":
        return base + (
            " Produce an extended meeting summary with sections in this order:\n"
            "<b>Контекст та мета</b>\n<b>Ключові рішення</b>\n<b>Завдання та відповідальність</b>\n"
            "<b>Ризики/Блокери</b>\n<b>Наступні кроки</b>\n"
            "Each section must start with the bold heading line, followed by bullets (• ...). "
            "Action items include Owner and <b>Дедлайн:</b> when present. Max ~600 words."
        )
    # minutes
    return base + (
        " Write formal meeting minutes with sections "
        "<b>Учасники</b>\n<b>Порядок денний / Обговорення</b>\n<b>Рішення</b>\n<b>Завдання</b>\n"
        "For tasks use one bullet per task: • Owner — Task. <b>Дедлайн:</b> date if known."
    )

async def generate_summary_mode(text: str, mode: str, target_lang: str) -> str:
    chunks = chunk_text(text, max_chars=8000)
    sys_msg = {"role": "system", "content": sys_prompt_for(mode, target_lang)}
    if len(chunks) == 1:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[sys_msg, {"role":"user","content":chunks[0]}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    # multi-pass
    partials = []
    for c in chunks:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[sys_msg, {"role":"user","content":c}],
            temperature=0.2
        )
        partials.append(res.choices[0].message.content.strip())
    glue = "\n\n---\n\n".join(partials)
    res2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[sys_msg, {"role":"user","content":f"Merge these sections into one coherent output:\n{glue}"}],
        temperature=0.2
    )
    return res2.choices[0].message.content.strip()

# -----------------------------------------------------------------------------
# Telegram HTML chunking / sending
# -----------------------------------------------------------------------------
def split_for_tg_html(html_text: str) -> List[str]:
    parts, cur = [], ""
    for line in html_text.splitlines(True):
        if len(cur) + len(line) > MAX_TG_HTML and cur:
            parts.append(cur); cur = ""
        cur += line
    if cur:
        parts.append(cur)
    return parts

async def send_long_html(chat_id: int, html_text: str, context: ContextTypes.DEFAULT_TYPE):
    for chunk in split_for_tg_html(html_text):
        await context.bot.send_message(chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

# -----------------------------------------------------------------------------
# Subtitles (SRT) + fallbacks
# -----------------------------------------------------------------------------
def _fmt_ts(sec: float) -> str:
    if sec < 0: sec = 0
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def make_srt_from_words(words: List[Dict], words_per_caption: int = 8) -> str:
    cues = []
    chunk = []
    for w in words:
        # glue punctuation to previous token
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

    lines = []
    for i, c in enumerate(cues, 1):
        start = c[0]["start"]
        end = c[-1]["end"]
        text = " ".join(w["word"] for w in c)
        lines.append(str(i))
        lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

def words_from_segments(segments: List[Dict]) -> List[Dict]:
    words: List[Dict] = []
    for seg in segments or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = (seg.get("text") or "").strip()
        tokens = text.split()
        if not tokens:
            continue
        dur = max(0.001, end - start)
        per = dur / len(tokens)
        t = start
        for w in tokens:
            words.append({"start": t, "end": min(end, t+per), "word": w})
            t += per
    return words

def linear_words_from_text(text: str, total_sec: float) -> List[Dict]:
    tokens = (text or "").split()
    if not tokens or total_sec <= 0:
        return []
    per = total_sec / max(1, len(tokens))
    t = 0.0
    words: List[Dict] = []
    for w in tokens:
        start = t
        end = min(total_sec, t + per)
        words.append({"start": start, "end": end, "word": w})
        t = end
    return words

def build_subtitles_wpc_kb(transcript_id: str, context: ContextTypes.DEFAULT_TYPE) -> InlineKeyboardMarkup:
    default_n = context.chat_data.get("wpc_default", 8)
    opts = [4,6,8,10,12]
    row = [InlineKeyboardButton(str(n), callback_data=f"subtitle_wpc:{n}:{transcript_id}") for n in opts]
    quick = [InlineKeyboardButton(f"Швидко: {default_n}", callback_data=f"subtitle_wpc:{default_n}:{transcript_id}")]
    return InlineKeyboardMarkup([
        quick,
        row,
        [InlineKeyboardButton("Ввести число…", callback_data=f"subtitle_custom:{transcript_id}")],
    ])

# -----------------------------------------------------------------------------
# Keyboards for actions
# -----------------------------------------------------------------------------
def build_actions_kb(transcript_id: str, lang_hint: str) -> InlineKeyboardMarkup:
    flip = "uk" if lang_hint == "en" else "en"
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Підсумок (короткий)",  callback_data=f"summary:short:{transcript_id}"),
            InlineKeyboardButton("Підсумок (розширений)", callback_data=f"summary:long:{transcript_id}"),
        ],
        [
            InlineKeyboardButton("Протокол", callback_data=f"summary:minutes:{transcript_id}"),
            InlineKeyboardButton(f"Переклад → {flip}", callback_data=f"translate:{flip}:{transcript_id}"),
        ],
        [
            InlineKeyboardButton("TXT", callback_data=f"export_txt:{transcript_id}"),
            InlineKeyboardButton("Субтитри", callback_data=f"subtitle:{transcript_id}"),
        ]
    ])

def build_loading_kb(label: str = "Обробка…") -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(f"⏳ {label}", callback_data="noop")]])

# -----------------------------------------------------------------------------
# Bot handlers
# -----------------------------------------------------------------------------
@dataclass
class MediaInfo:
    filename: str
    language: Optional[str]  # from caption "lang=uk" | "en" | None

def parse_caption_language(caption: Optional[str]) -> Optional[str]:
    if not caption: return None
    for p in caption.split():
        if p.lower().startswith("lang="):
            val = p.split("=",1)[1].strip().lower()
            if val in ("uk","en"): return val
    return None

async def start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Надішли voice/audio/відео або файл — поверну текст.\n"
        "Можна додати підпис: lang=uk або lang=en.\n"
        "Після розшифровки з’являться кнопки: Підсумок, Протокол, Переклад, TXT, Субтитри."
    )

async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start — привітання\n"
        "Надішли аудіо/voice/відео/файл — отримаєш розшифровку.\n"
        "Порада: додай підпис lang=uk або lang=en.\n"
        "/settings — налаштування (мова підсумку, тип підсумку за замовчуванням)."
    )

async def show_settings_menu(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    st = context.chat_data.setdefault("settings", dict(DEFAULT_SETTINGS))
    sm = st.get("summary_mode","short")
    sm_label = {"short":"короткий", "long":"розширений", "minutes":"протокол"}.get(sm,"короткий")
    sl = st.get("summary_lang","auto")
    sl_label = {"auto":"Авто", "uk":"Українська", "en":"English"}[sl]
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Мова підсумку: {sl_label}", callback_data="settings:summary_lang:cycle")],
        [InlineKeyboardButton(f"Формат підсумку: {sm_label}", callback_data="settings:summary_mode:cycle")],
        [InlineKeyboardButton("Закрити", callback_data="settings:close")],
    ])
    await context.bot.send_message(chat_id, "Налаштування", reply_markup=kb)

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await show_settings_menu(update.effective_chat.id, context)

async def handle_settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    parts = query.data.split(":")  # settings:key:action
    if len(parts) < 3:
        return
    _, key, action = parts
    st = context.chat_data.setdefault("settings", dict(DEFAULT_SETTINGS))
    if key == "summary_lang" and action == "cycle":
        order = ["auto","uk","en"]
        cur = st.get("summary_lang","auto")
        st["summary_lang"] = order[(order.index(cur)+1) % len(order)]
    elif key == "summary_mode" and action == "cycle":
        order = ["short","long","minutes"]
        cur = st.get("summary_mode","short")
        st["summary_mode"] = order[(order.index(cur)+1) % len(order)]
    elif key == "close":
        try:
            await query.delete_message()
        except Exception:
            pass
        return
    await query.edit_message_reply_markup(reply_markup=None)
    await show_settings_menu(update.effective_chat.id, context)

# -----------------------------------------------------------------------------
# Media handling with visible statuses and long-file split
# -----------------------------------------------------------------------------
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

    status = await msg.reply_text("🎧 Отримав файл — готую аудіо…")
    try:
        await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

        # determine file & name
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
            await status.edit_text("Надішліть аудіо/voice/відео/файл.")
            return

        language = parse_caption_language(msg.caption)  # None/uk/en

        if file.file_size and file.file_size > MAX_API_FILE_BYTES * 8:  # soft limit
            await status.edit_text(f"Файл надто великий ({human_size(file.file_size)}). Надішліть менший.")
            return

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / filename
            await file.download_to_drive(custom_path=str(src))

            await status.edit_text("🎚️ Конвертую у WAV…")
            wav = td / TMP_WAV
            try:
                # run ffmpeg conversion in a background thread to avoid blocking the event loop
                await asyncio.to_thread(run_ffmpeg_to_wav, src, wav, sr=AUDIO_SR)
            except subprocess.CalledProcessError:
                await status.edit_text("❌ Не вдалося обробити аудіо через ffmpeg.")
                return

            # compute duration in a thread so large files don't block
            duration_sec = await asyncio.to_thread(media_duration_seconds, wav)

            if duration_sec > SPLIT_THRESHOLD_SEC:
                await status.edit_text("✂️ Довгий файл — ділю на частини…")
                # perform splitting in background thread
                chunks = await asyncio.to_thread(split_wav_to_chunks, wav, td, chunk_sec=SPLIT_CHUNK_SEC)
                total_text = []
                all_words: List[Dict] = []
                all_segs: List[Dict] = []
                offset = 0.0
                for i, ch in enumerate(chunks, 1):
                    await status.edit_text(f"🧠 Транскрибую… (частина {i}/{len(chunks)})")
                    # compress each chunk in background thread
                    candidate = await asyncio.to_thread(ensure_under_limit, ch)
                    r = await asyncio.to_thread(transcribe_verbose, candidate, language)
                    t = (r.get("text") or "").strip()
                    if t:
                        total_text.append(t)
                    for w in r.get("words") or []:
                        all_words.append({"start": w["start"] + offset, "end": w["end"] + offset, "word": w["word"]})
                    for s in r.get("segments") or []:
                        all_segs.append({"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"]})
                    # measure chunk duration in thread
                    ch_dur = await asyncio.to_thread(media_duration_seconds, ch)
                    offset += ch_dur if ch_dur > 0 else (all_words[-1]["end"] - all_words[0]["start"] if all_words else 0)
                result = {"text": "\n".join(total_text).strip(), "words": all_words, "segments": all_segs, "lang": None}
            else:
                await status.edit_text("🗜️ Стискаю до ліміту…")
                # ensure file is under OpenAI limit in background thread
                candidate = await asyncio.to_thread(ensure_under_limit, wav)
                await status.edit_text("🧠 Транскрибую…")
                result = await asyncio.to_thread(transcribe_verbose, candidate, language)

            text = (result.get("text") or "").strip()
            if not text:
                await status.edit_text("❌ Не вдалось розпізнати мовлення (порожній результат).")
                return

            # Store transcript + words + segments + duration
            transcripts: Dict[str, str] = context.chat_data.setdefault("transcripts", {})
            words_map: Dict[str, List[Dict]] = context.chat_data.setdefault("transcript_words", {})
            segs_map: Dict[str, List[Dict]] = context.chat_data.setdefault("transcript_segments", {})
            dur_map: Dict[str, float] = context.chat_data.setdefault("transcript_durations", {})

            transcript_id = uuid.uuid4().hex
            transcripts[transcript_id] = text
            segs_map[transcript_id] = result.get("segments") or []
            words_map[transcript_id] = result.get("words") or []
            dur_map[transcript_id] = duration_sec

            # Language hint
            lang_hint = "uk"
            lang_det = result.get("lang")
            if lang_det:
                lang_hint = "uk" if str(lang_det).lower().startswith("uk") else "en"
            context.chat_data["last_transcript_lang"] = lang_hint
            context.chat_data["last_lang_hint"] = lang_hint

            await status.edit_text("✅ Готово! Відправляю результати…")
            # Send text or file depending on length
            if len(text) > 3500:
                txt = td / "transcript.txt"
                txt.write_text(text, encoding="utf-8")
                await msg.reply_document(
                    document=InputFile(str(txt), filename="transcript.txt"),
                    caption="Розшифровка",
                    reply_markup=build_actions_kb(transcript_id, lang_hint)
                )
            else:
                await msg.reply_text(text, reply_markup=build_actions_kb(transcript_id, lang_hint))
    finally:
        try:
            await status.delete()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Callback handling
# -----------------------------------------------------------------------------
async def handle_misc_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data == "noop":
        return await query.answer("Обробляю…")
    if data.startswith("settings:"):
        return await handle_settings_callback(update, context)

    parts = data.split(":")
    action = parts[0]
    chat_id = update.effective_chat.id
    transcripts: Dict[str, str] = context.chat_data.get("transcripts", {})
    words_map: Dict[str, List[Dict]] = context.chat_data.get("transcript_words", {})
    segs_map: Dict[str, List[Dict]] = context.chat_data.get("transcript_segments", {})
    dur_map: Dict[str, float] = context.chat_data.get("transcript_durations", {})
    lang_hint = context.chat_data.get("last_lang_hint", "uk")

    if action == "summary":
        # summary:<mode>:<tid>
        mode = parts[1]; tid = parts[2]
        text = transcripts.get(tid, "")
        if not text:
            return await query.answer("Немає тексту для підсумку", show_alert=True)
        await query.answer("Готую підсумок…")
        await query.edit_message_reply_markup(reply_markup=build_loading_kb(
            "Підсумок (розширений…)" if mode=="long" else "Протокол…" if mode=="minutes" else "Підсумок…"
        ))
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        st = context.chat_data.get("settings", DEFAULT_SETTINGS)
        target = pick_output_lang(context.chat_data.get("last_transcript_lang"), st)
        summary = await generate_summary_mode(text, mode, target)
        title = "Підсумок (long)" if mode=="long" else ("Протокол" if mode=="minutes" else "Підсумок (short)")
        html_out = f"<b>{title}</b>\n{summary}"
        await send_long_html(chat_id, html_out, context)
        await query.edit_message_reply_markup(reply_markup=build_actions_kb(tid, target))
        return

    if action == "translate":
        await query.answer("Поки не реалізовано переклад у цій збірці", show_alert=True)
        return

    if action == "export_txt":
        tid = parts[1]
        text = transcripts.get(tid, "")
        if not text:
            return await query.answer("Немає тексту", show_alert=True)
        await query.answer("Генерую TXT…")
        await query.edit_message_reply_markup(reply_markup=build_loading_kb("TXT…"))
        from io import BytesIO
        bio = BytesIO(text.encode("utf-8")); bio.name = "transcript.txt"
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)
        await context.bot.send_document(chat_id, document=InputFile(bio), caption="Транскрипт (TXT)")
        await query.edit_message_reply_markup(reply_markup=build_actions_kb(tid, lang_hint))
        return

    if action == "subtitle":
        tid = parts[1]

        # Build robust words list: words -> segments -> linear
        words = words_map.get(tid, [])
        if not words:
            segs = segs_map.get(tid, [])
            if segs:
                words = words_from_segments(segs)
                words_map[tid] = words
            else:
                txt = transcripts.get(tid, "")
                dur = dur_map.get(tid, 0.0)
                approx = linear_words_from_text(txt, dur)
                if approx:
                    words = approx
                    words_map[tid] = words

        if not words:
            return await query.answer("Не вдалося створити таймкоди. Спробуйте інший файл.", show_alert=True)

        await query.answer("Скільки слів на таймкод?")
        await query.edit_message_reply_markup(reply_markup=build_subtitles_wpc_kb(tid, context))
        return

    if action == "subtitle_wpc":
        n = int(parts[1]); tid = parts[2]
        words = words_map.get(tid, [])
        if not words:
            return await query.answer("Немає таймкодів", show_alert=True)
        await query.answer("Генерую SRT…")
        await query.edit_message_reply_markup(reply_markup=build_loading_kb("Субтитри…"))
        from io import BytesIO
        srt = make_srt_from_words(words, n)
        bio = BytesIO(srt.encode("utf-8")); bio.name = f"subtitles_{n}w.srt"
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)
        await context.bot.send_document(chat_id, document=InputFile(bio), caption=f"Субтитри ({n} слів/таймкод)")
        context.chat_data["wpc_default"] = n  # remember last choice
        await query.edit_message_reply_markup(reply_markup=build_actions_kb(tid, lang_hint))
        return

    if action == "subtitle_custom":
        tid = parts[1]
        context.chat_data["awaiting_wpc_for"] = tid
        await query.answer()
        await context.bot.send_message(chat_id, "Введи кількість слів на таймкод (1–30):")
        return

async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    tid = context.chat_data.get("awaiting_wpc_for")
    if not tid:
        return
    try:
        n = int(msg.text.strip())
        if not (1 <= n <= 30):
            raise ValueError
    except Exception:
        await msg.reply_text("Будь ласка, введи число від 1 до 30.")
        return

    words_map: Dict[str, List[Dict]] = context.chat_data.get("transcript_words", {})
    words = words_map.get(tid, [])
    if not words:
        await msg.reply_text("Немає таймкодів для цього запису.")
        context.chat_data.pop("awaiting_wpc_for", None)
        return

    from io import BytesIO
    srt = make_srt_from_words(words, n)
    bio = BytesIO(srt.encode("utf-8")); bio.name = f"subtitles_{n}w.srt"
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.UPLOAD_DOCUMENT)
    await msg.reply_document(document=InputFile(bio), caption=f"Субтитри ({n} слів/таймкод)")
    context.chat_data["wpc_default"] = n  # remember last choice
    context.chat_data.pop("awaiting_wpc_for", None)
    # Return menu for convenience
    lang_hint = context.chat_data.get("last_lang_hint", "uk")
    transcripts: Dict[str, str] = context.chat_data.get("transcripts", {})
    # find any transcript containing this tid to attach keyboard correctly; here we just reuse tid
    await msg.reply_text("Готово. Обери наступну дію ↓", reply_markup=build_actions_kb(tid, lang_hint))

# -----------------------------------------------------------------------------
# Wiring & main
# -----------------------------------------------------------------------------
def build_application() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    app.add_handler(CallbackQueryHandler(handle_misc_callback))
    # media + plain text
    app.add_handler(MessageHandler(AUDIO_FILTER, handle_media))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_plain_text))
    return app

def main():
    app = build_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
