import io
import os
import re
import subprocess
import uuid

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from demater import DeMater

app = FastAPI(title="DeMater Web Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация движков: Vosk и Whisper Fast активны, стандартный Transformers Whisper отключен
model_path = os.environ.get("DEMATBOT_MODEL_PATH", "models/vosk-model-small-ru-0.22")
demater = DeMater(
    vosk_model_path=model_path,
    enable_vosk=True,
    enable_faster_whisper=True,
    enable_whisper=False,
)

# Обычный in-memory словарь: файлы удаляются сразу после воспроизведения/скачивания
audio_cache: dict[str, bytes] = {}

def markdown_spoiler_to_html(text: str) -> str:
    cleaned = text.replace(r"\|", "|")
    cleaned = re.sub(r"\\([-_*\[\]()~`>#+=|{}.!])", r"\1", cleaned)
    return re.sub(
        r"\|\|(.*?)\|\|",
        r'<span class="spoiler" title="Нажмите, чтобы показать">\1</span>',
        cleaned,
    )


def convert_audio_to_wav(input_bytes: bytes) -> io.BytesIO:
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-f", "wav",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-vn",
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    audio_file: UploadFile = File(None),
):
    # --- ОБРАБОТКА АУДИО ---
    if audio_file:
        content = await audio_file.read()
        try:
            wav_buffer = await run_in_threadpool(convert_audio_to_wav, content)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка конвертации аудио: {str(e)}"},
            )

        targetwords = demater.get_target_word_list_or_default(session_id=session_id)

        # Выполняем в тредпуле, чтобы не блокировать FastAPI Event Loop
        result = await run_in_threadpool(
            demater.process,
            input_file=wav_buffer,
            target_words=targetwords,
            session_id=session_id,
        )

        audio_id = str(uuid.uuid4())
        audio_cache[audio_id] = result["out_file"].getvalue()

        # Динамическое формирование HTML только для включенных движков
        blocks = []
        for _, method_info in result.get("methods", {}).items():
            name = method_info["name"]
            masked = markdown_spoiler_to_html(method_info["masked_text"])
            count = method_info["count"]
            blocks.append(
                f"<b>Вариант ({name}):</b><br/>{masked}<br/><i>Найдено слов: {count}</i>"
            )

        formatted_msg = "<br/><br/>".join(blocks) if blocks else "Распознавание завершено."

        return {
            "type": "audio",
            "text_html": formatted_msg,
            "audio_url": f"/api/audio/{audio_id}",
        }

    # --- ОБРАБОТКА ТЕКСТА И КОМАНД ---
    text = (text or "").strip()
    if not text:
        return {"type": "text", "text_html": "Пустое сообщение"}

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
            ),
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

    targetwords_raw = demater.get_target_word_list_or_default(session_id=session_id)
    targetwords = [w for w in targetwords_raw.split(",") if w.strip()]
    replaced = demater.replace_text(text, targetwords)

    return {
        "type": "text",
        "text_html": markdown_spoiler_to_html(replaced),
    }


@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str):
    # .pop удаляет элемент из словаря и сразу освобождает память под аудио
    audio_data = audio_cache.pop(audio_id, None)
    if audio_data is None:
        return JSONResponse(
            status_code=404, 
            content={"error": "Audio not found or already consumed"}
        )
    return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()