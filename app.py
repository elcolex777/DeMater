import io
import os
import re
import subprocess
import time
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

# Конфигурация движков
model_path = os.environ.get("DEMATBOT_MODEL_PATH", "models/vosk-model-small-ru-0.22")
demater = DeMater(
    vosk_model_path=model_path,
    enable_vosk=True,
    enable_faster_whisper=True,
    enable_whisper=False,
)

# Кэш аудио с временными метками для возможности повторного скачивания/прослушивания
# Структура: {audio_id: (bytes_data, timestamp)}
audio_cache: dict[str, tuple[bytes, float]] = {}
CACHE_TTL_SECONDS = 3600  # Хранить файлы в течение 1 часа


def cleanup_cache():
    now = time.time()
    expired = [k for k, (_, ts) in audio_cache.items() if now - ts > CACHE_TTL_SECONDS]
    for k in expired:
        audio_cache.pop(k, None)


def markdown_spoiler_to_html(text: str) -> str:
    cleaned = text.replace(r"\|", "|")
    cleaned = re.sub(r"\\([-_*\[\]()~`>#+=|{}.!])", r"\1", cleaned)
    return re.sub(
        r"\|\|(.*?)\|\|",
        r'<span class="spoiler" title="Нажмите, чтобы показать">\1</span>',
        cleaned,
    )


def convert_audio_to_wav(input_bytes: bytes) -> io.BytesIO:
    """Извлекает и конвертирует аудио (включая видео-контейнеры) в 16kHz mono WAV."""
    cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-vn",                 # Игнорировать видеопоток
        "-f", "wav",
        "-ar", "16000",        # Частота 16 kHz для Whisper/Vosk
        "-ac", "1",            # Моно
        "-acodec", "pcm_s16le",
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
    cleanup_cache()

    # --- ОБРАБОТКА АУДИО И ВИДЕО ---
    if audio_file:
        content = await audio_file.read()
        try:
            # Конвертируем входные байты (аудио или видео) в нужный WAV-формат
            wav_buffer = await run_in_threadpool(convert_audio_to_wav, content)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка обработки медиафайла: {str(e)}"},
            )

        # Сохраняем оригинальный нормализованный звук для воспроизведения
        original_audio_id = str(uuid.uuid4())
        audio_cache[original_audio_id] = (wav_buffer.getvalue(), time.time())
        wav_buffer.seek(0)

        targetwords = demater.get_target_word_list_or_default(session_id=session_id)

        # Выполняем цензурирование
        result = await run_in_threadpool(
            demater.process,
            input_file=wav_buffer,
            target_words=targetwords,
            session_id=session_id,
        )

        processed_audio_id = str(uuid.uuid4())
        audio_cache[processed_audio_id] = (result["out_file"].getvalue(), time.time())

        # Формирование текстового отчета
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
            "audio_url": f"/api/audio/{processed_audio_id}",
            "original_audio_url": f"/api/audio/{original_audio_id}",
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
                "Вы можете отправить текст, загрузить аудио/видеофайл или записать голосовое сообщение прямо с микрофона.<br/><br/>"
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


from fastapi import Request, Response, status

@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str, request: Request):
    item = audio_cache.get(audio_id)
    if item is None:
        return JSONResponse(
            status_code=404, 
            content={"error": "Audio not found or expired"}
        )
    audio_data, _ = item
    file_size = len(audio_data)
    range_header = request.headers.get("range")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"inline; filename={audio_id}.wav",
    }

    # Если клиент (Safari/iPad) запросил диапазон байт
    if range_header:
        try:
            # Формат заголовка: "bytes=start-end"
            range_value = range_header.strip().split("=")[-1]
            start_str, end_str = range_value.split("-")
            
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            
            chunk_length = end - start + 1
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(chunk_length)
            
            return Response(
                content=audio_data[start : end + 1],
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                headers=headers,
                media_type="audio/wav",
            )
        except Exception:
            pass

    # Обычный ответ, если Range не передан
    headers["Content-Length"] = str(file_size)
    return Response(
        content=audio_data,
        status_code=status.HTTP_200_OK,
        headers=headers,
        media_type="audio/wav",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()