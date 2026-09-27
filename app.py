import io
import os
import uuid
import subprocess
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import re
from demater import DeMater

app = FastAPI(title="DeMater Web Chat")

# Разрешаем CORS при необходимости
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.environ.get('DEMATBOT_MODEL_PATH', "models/vosk-model-small-ru-0.22")
demater = DeMater(model_path=model_path)

# Временное in-memory хранилище обработанных аудиофайлов для отдачи в плеер
audio_cache: dict[str, bytes] = {}

def clean_markdown_escapes(text: str) -> str:
    """Удаляет слэши экранирования Telegram MarkdownV2 для вывода в HTML."""
    return text.replace(r'\_', '_').replace(r'\*', '*').replace(r'\[', '[').replace(r'\]', ']') \
               .replace(r'\(', '(').replace(r'\)', ')').replace(r'\~', '~').replace(r'\`', '`') \
               .replace(r'\>', '>').replace(r'\#', '#').replace(r'\+', '+').replace(r'\-', '-') \
               .replace(r'\=', '=').replace(r'\|', '|').replace(r'\{', '{').replace(r'\}', '}') \
               .replace(r'\.', '.').replace(r'\!', '!')


def markdown_spoiler_to_html(text: str) -> str:
    """Очищает экранирование Telegram MarkdownV2 и преобразует ||текст|| в спойлер."""
    # demater.py оборачивает слова в ||слово||, а затем экранирует все спецсимволы в \|
    # Поэтому маркер может выглядеть как \||слово\|| или \|\|слово\|\|
    # 1. Приводим экранированные вертикальные черты к обычному виду
    cleaned = text.replace(r'\|', '|')
    # 2. Убираем остальные слеши экранирования Telegram MarkdownV2
    cleaned = re.sub(r'\\([-_*\[\]()~`>#+=|{}.!])', r'\1', cleaned)
    # 3. Преобразуем ||...|| в интерактивный span
    return re.sub(
        r'\|\|(.*?)\|\|',
        r'<span class="spoiler" title="Нажмите, чтобы показать">\1</span>',
        cleaned
    )

def convert_audio_to_wav(input_bytes: bytes) -> io.BytesIO:
    """
    Конвертирует аудио любых форматов (webm, ogg, mp4, aac, mp3, flac)
    в 16-битный PCM WAV моно с частотой 16 кГц через ffmpeg в памяти.
    """
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",           # Чтение из stdin
        "-f", "wav",              # Формат вывода: WAV
        "-ar", "16000",           # Частота дискретизации: 16000 Гц
        "-ac", "1",               # Моноканал
        "-acodec", "pcm_s16le",   # 16-битный PCM (для vosk и wave.open)
        "-vn",                    # Игнорировать видеоряд, если он есть
        "pipe:1"                  # Запись в stdout
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = process.communicate(input=input_bytes)

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode(errors='ignore')}")

    wav_buffer = io.BytesIO(out)
    wav_buffer.seek(0)
    return wav_buffer


@app.post("/api/message")
async def handle_message(
    session_id: str = Form(...),
    text: str = Form(None),
    audio_file: UploadFile = File(None)
):
    # --- ОБРАБОТКА АУДИО ---
    if audio_file:
        content = await audio_file.read()
        try:
            # Преобразуем входящий аудиопоток через ffmpeg
            wav_buffer = convert_audio_to_wav(content)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка конвертации аудио: {str(e)}"}
            )

        targetwords = demater.get_target_word_list_or_default(session_id=session_id)
        result = demater.process(input_file=wav_buffer, target_words=targetwords, session_id=session_id)

        audio_id = str(uuid.uuid4())
        audio_cache[audio_id] = result["out_file"].getvalue()

        formatted_msg = (
            f"<b>Вариант 1 (Vosk):</b><br/>{markdown_spoiler_to_html(result['text'])}<br/>"
            f"<i>Матерных слов: {result['detected_word_list_count']}</i><br/><br/>"
            f"<b>Вариант 2 (Whisper):</b><br/>{markdown_spoiler_to_html(result['text_whisper'])}<br/>"
            f"<i>Матерных слов: {result['detected_word_list2_count']}</i>"
        )

        return {
            "type": "audio",
            "text_html": formatted_msg,
            "audio_url": f"/api/audio/{audio_id}"
        }

    # --- ОБРАБОТКА ТЕКСТА И КОМАНД ---
    text = (text or "").strip()
    if not text:
        return {"type": "text", "text_html": "Пустое сообщение"}

    # Эмуляция команд бота
    if text == "/start":
        return {
            "type": "text",
            "text_html": (
                "<b>Привет!</b><br/>"
                "Этот сервис запикивает части аудио с матом.<br/>"
                "Вы можете отправить текст, загрузить аудиофайл или записать голосовое сообщение прямо с микрофона.<br/><br/>"
                "<b>Доступные команды:</b><br/>"
                "<code>/targetwords</code> — посмотреть список слов<br/>"
                "<code>/targetwords_set слово1 слово2</code> — задать свой список слов<br/>"
                "<code>/targetwords_add слово1 слово2</code> — добавить слова в список<br/>"
                "<code>/targetwords_reset</code> — сбросить список к стандартному"
            )
        }

    if text.startswith("/targetwords_reset"):
        if session_id in demater.user_data:
            demater.user_data[session_id]["target_word_list_custom"] = ""
        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        return {"type": "text", "text_html": f"Список сброшен:<br/>{markdown_spoiler_to_html(masked)}..."}

    if text.startswith("/targetwords_set"):
        words = text.replace("/targetwords_set", "").strip().replace(" ", ",").lower()
        if words:
            demater.get_user_data_or_new(session_id)["target_word_list_custom"] = words
        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        return {"type": "text", "text_html": f"Установлен список:<br/>{markdown_spoiler_to_html(masked)}"}

    if text.startswith("/targetwords_add"):
        words = text.replace("/targetwords_add", "").strip().replace(" ", ",").lower()
        if words:
            current = demater.get_target_word_list_or_default(session_id=session_id)
            demater.get_user_data_or_new(session_id)["target_word_list_custom"] = f"{words},{current}"
        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        return {"type": "text", "text_html": f"Слова добавлены. Текущий список:<br/>{markdown_spoiler_to_html(masked)}"}

    if text.startswith("/targetwords"):
        words = text.replace("/targetwords", "").strip().replace(" ", ",").lower()
        if words:
            demater.get_user_data_or_new(session_id)["target_word_list_custom"] = words
        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        suffix = "..." if len(targetwords) >= 20 else ""
        return {"type": "text", "text_html": f"Текущие слова:<br/>{markdown_spoiler_to_html(masked)}{suffix}"}

    # Обычный текст (фильтрация мата)
    targetwords_raw = demater.get_target_word_list_or_default(session_id=session_id)
    targetwords = [w for w in targetwords_raw.split(",") if w.strip()]
    replaced = demater.replace_text(text, targetwords)

    return {
        "type": "text",
        "text_html": markdown_spoiler_to_html(replaced)
    }


@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str):
    if audio_id not in audio_cache:
        return JSONResponse(status_code=404, content={"error": "Audio not found"})
    return StreamingResponse(io.BytesIO(audio_cache[audio_id]), media_type="audio/wav")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()