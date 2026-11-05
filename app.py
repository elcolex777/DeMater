import asyncio
import io
import json
import logging
import os
import re
import subprocess
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, Request, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from demater import DeMater

logger = logging.getLogger("app")

app = FastAPI(title="DeMater Web Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.environ.get("DEMATBOT_MODEL_PATH", "models/vosk-model-small-ru-0.22")
demater = DeMater(
    vosk_model_path=model_path,
    enable_vosk=True,
    enable_faster_whisper=True,
    enable_whisper=False,
)

audio_cache: dict[str, dict] = {}

ACCESS_GRACE_PERIOD_SEC = 20 * 60
UNACCESSED_TTL_SEC = 1800


def cleanup_audio_cache():
    now = time.time()
    expired_ids = []
    for audio_id, item in audio_cache.items():
        if item["first_accessed_at"] is not None:
            if now - item["first_accessed_at"] > ACCESS_GRACE_PERIOD_SEC:
                expired_ids.append(audio_id)
        elif now - item["created_at"] > UNACCESSED_TTL_SEC:
            expired_ids.append(audio_id)

    for audio_id in expired_ids:
        audio_cache.pop(audio_id, None)


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
        "-y",
        "-i", "pipe:0",
        "-vn",
        "-f", "wav",
        "-ar", "16000",
        "-ac", "1",
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


@app.get("/api/sounds")
async def list_sounds(session_id: str):
    sounds = demater.get_available_sounds(session_id=session_id)
    return {"sounds": sounds}


@app.post("/api/sounds/select")
async def select_sound(session_id: str = Form(...), sound_id: str = Form(...)):
    ok = demater.set_user_sound(session_id, sound_id)
    if not ok:
        return JSONResponse(status_code=400, content={"error": "Звук не найден"})
    return {"status": "ok", "selected_sound_id": sound_id}


@app.post("/api/sounds/upload")
async def upload_sound(
    session_id: str = Form(...),
    name: str = Form(...),
    sound_file: UploadFile = File(...),
):
    content = await sound_file.read()
    try:
        sound_id = await run_in_threadpool(demater.register_custom_sound, name, content)
        demater.set_user_sound(session_id, sound_id)
        return {"status": "ok", "sound_id": sound_id, "name": name}
    except Exception as e:
        logger.exception("Ошибка сохранения звука")
        return JSONResponse(status_code=400, content={"error": f"Ошибка сохранения звука: {str(e)}"})


@app.get("/api/sounds/{sound_id}/preview")
async def preview_sound(sound_id: str):
    if sound_id == "silence":
        return Response(content=b"", media_type="audio/wav")

    file_path = demater.custom_beeps_dir / f"{sound_id}.wav"
    if not file_path.exists():
        if sound_id == "default":
            file_path_str = os.path.join("data", "censor-beep-2-16000.wav")
            if not os.path.exists(file_path_str):
                file_path_str = os.path.join("data", "censor-beep-2-2.wav")
            if not os.path.exists(file_path_str):
                return Response(content=b"\x00" * 32000, media_type="audio/wav")
            with open(file_path_str, "rb") as f:
                return Response(content=f.read(), media_type="audio/wav")
        return JSONResponse(status_code=404, content={"error": "Файл не найден"})

    return Response(content=file_path.read_bytes(), media_type="audio/wav")


def build_text_command_response(text: str, session_id: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"type": "text", "text_html": "Пустое сообщение"}

    if text == "/start":
        return {
            "type": "text",
            "text_html": (
                "<b>Привет!</b><br/>"
                "Этот сервис запикивает части аудио с нецензурной лексикой.<br/>"
                "Вы можете отправить текст, загрузить аудио/видеофайл или записать голосовое сообщение прямо с микрофона.<br/><br/>"
                "<b>Доступные команды:</b><br/>"
                "<code>/targetwords</code> — посмотреть список слов<br/>"
                "<code>/targetwords_set слово1 слово2</code> — задать свой список слов<br/>"
                "<code>/targetwords_add слово1 слово2</code> — добавить слова в список<br/>"
                "<code>/targetwords_reset</code> — сбросить список к стандартному<br/>"
                "<code>/sounds</code> — список звуков запикивания<br/>"
                "<code>/sound_set &lt;id&gt;</code> — переключить звук (например: <code>/sound_set silence</code>)<br/>"
                "<code>/sound_reset</code> — вернуть стандартный пик<br/>"
                "<code>/beep_margins 0.07 0.07</code> — отступы оригинального звука (начало/конец в сек)<br/>"
                "<code>/beep_short 0.210 0.250</code> — порог короткого слова и длина запикивания<br/>"
                "<code>/beep_max 0.5</code> — максимальная длительность запикивания слова"
            ),
        }

    if text.startswith("/sounds"):
        available = demater.get_available_sounds(session_id=session_id)
        items = []
        for s in available:
            flag = "👉 <b>[Выбран]</b> " if s["is_active"] else ""
            items.append(f"• {flag}<code>/sound_set {s['id']}</code> — {s['name']}")
        msg = "<b>Доступные звуки запикивания:</b><br/>" + "<br/>".join(items)
        msg += "<br/><br/><i>Для возврата к стандарту используйте: <code>/sound_reset</code></i>"
        return {"type": "text", "text_html": msg}

    if text.startswith("/sound_set"):
        snd_id = text.replace("/sound_set", "").strip()
        if demater.set_user_sound(session_id, snd_id):
            return {"type": "text", "text_html": f"Звук запикивания изменен на: <b>{snd_id}</b>"}
        return {"type": "text", "text_html": f"Звук <code>{snd_id}</code> не найден. Введите <code>/sounds</code> для списка."}

    if text.startswith("/sound_reset"):
        demater.set_user_sound(session_id, "default")
        return {"type": "text", "text_html": "Установлен стандартный пик."}

    if text.startswith("/beep_margins"):
        parts = text.split()
        user_dict = demater.get_user_data_or_new(session_id)
        if len(parts) >= 3:
            try:
                user_dict["lead_in_sec"] = float(parts[1])
                user_dict["lead_out_sec"] = float(parts[2])
                return {
                    "type": "text",
                    "text_html": f"Установлены отступы: в начале <b>{parts[1]} с</b>, в конце <b>{parts[2]} с</b>.",
                }
            except ValueError:
                pass
        curr_in = user_dict.get("lead_in_sec", 0.07)
        curr_out = user_dict.get("lead_out_sec", 0.07)
        return {
            "type": "text",
            "text_html": (
                f"Текущие отступы: в начале <b>{curr_in} с</b>, в конце <b>{curr_out} с</b>.<br/>"
                "Формат команды: <code>/beep_margins 0.07 0.07</code>"
            ),
        }

    if text.startswith("/beep_short"):
        parts = text.split()
        user_dict = demater.get_user_data_or_new(session_id)
        if len(parts) >= 3:
            try:
                user_dict["short_word_threshold"] = float(parts[1])
                user_dict["short_word_beep_duration"] = float(parts[2])
                return {
                    "type": "text",
                    "text_html": f"Настройки коротких слов сохранены: порог <b>{parts[1]} с</b>, длина запикивания <b>{parts[2]} с</b>.",
                }
            except ValueError:
                pass
        curr_th = user_dict.get("short_word_threshold", 0.210)
        curr_dur = user_dict.get("short_word_beep_duration", 0.250)
        return {
            "type": "text",
            "text_html": (
                f"Текущие настройки коротких слов: порог <b>{curr_th} с</b>, длина запикивания <b>{curr_dur} с</b>.<br/>"
                "Формат команды: <code>/beep_short 0.210 0.250</code>"
            ),
        }

    if text.startswith("/beep_max"):
        parts = text.split()
        user_dict = demater.get_user_data_or_new(session_id)
        if len(parts) >= 2:
            try:
                user_dict["max_word_beep_duration"] = float(parts[1])
                return {
                    "type": "text",
                    "text_html": f"Максимальная длина запикивания установлена на: <b>{parts[1]} с</b>.",
                }
            except ValueError:
                pass
        curr_max = user_dict.get("max_word_beep_duration", 0.500)
        return {
            "type": "text",
            "text_html": (
                f"Текущий максимум длины запикивания: <b>{curr_max} с</b>.<br/>"
                "Формат команды: <code>/beep_max 0.5</code>"
            ),
        }

    if text.startswith("/targetwords_reset"):
        if session_id in demater.user_data:
            demater.user_data[session_id]["target_word_list_custom"] = ""
        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        return {"type": "text", "text_html": f"Список сброшен:<br/>{markdown_spoiler_to_html(masked)}..."}

    if text.startswith("/targetwords_set"):
        words_raw = text.replace("/targetwords_set", "").strip()
        parsed_words = [w.strip().lower() for w in re.split(r"[\s,]+", words_raw) if w.strip()]
        if parsed_words:
            demater.save_user_suggested_words(parsed_words)
            demater.get_user_data_or_new(session_id)["target_word_list_custom"] = ",".join(parsed_words)

        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        return {"type": "text", "text_html": f"Установлен список:<br/>{markdown_spoiler_to_html(masked)}"}

    if text.startswith("/targetwords_add"):
        words_raw = text.replace("/targetwords_add", "").strip()
        parsed_words = [w.strip().lower() for w in re.split(r"[\s,]+", words_raw) if w.strip()]
        if parsed_words:
            demater.save_user_suggested_words(parsed_words)
            current = demater.get_target_word_list_or_default(session_id=session_id)
            new_joined = ",".join(parsed_words)
            demater.get_user_data_or_new(session_id)["target_word_list_custom"] = f"{new_joined},{current}"

        targetwords = demater.get_target_word_list_or_default(session_id=session_id).split(",")[:20]
        masked = demater.replace_text(" ".join(targetwords), targetwords)
        return {"type": "text", "text_html": f"Слова добавлены. Текущий список:<br/>{markdown_spoiler_to_html(masked)}"}

    if text.startswith("/targetwords"):
        words_raw = text.replace("/targetwords", "").strip()
        parsed_words = [w.strip().lower() for w in re.split(r"[\s,]+", words_raw) if w.strip()]
        if parsed_words:
            demater.save_user_suggested_words(parsed_words)
            demater.get_user_data_or_new(session_id)["target_word_list_custom"] = ",".join(parsed_words)

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


@app.post("/api/message/stream")
async def handle_message_stream(
    session_id: str = Form(...),
    text: str = Form(None),
    audio_file: UploadFile = File(None),
    lead_in: Optional[float] = Form(None),
    lead_out: Optional[float] = Form(None),
    short_thresh: Optional[float] = Form(None),
    short_beep: Optional[float] = Form(None),
    max_beep: Optional[float] = Form(None),
):
    cleanup_audio_cache()

    audio_bytes: Optional[bytes] = None
    if audio_file:
        audio_bytes = await audio_file.read()

    async def event_generator():
        if not audio_bytes:
            res_data = build_text_command_response(text, session_id)
            yield f"event: final\ndata: {json.dumps(res_data, ensure_ascii=False)}\n\n"
            return

        try:
            wav_buffer = await run_in_threadpool(convert_audio_to_wav, audio_bytes)
        except Exception as e:
            logger.exception("Ошибка конвертации аудиофайла")
            err_data = {"error": f"Ошибка конвертации медиа: {str(e) or 'некорректный формат'}"}
            yield f"event: final\ndata: {json.dumps(err_data, ensure_ascii=False)}\n\n"
            return

        original_audio_id = str(uuid.uuid4())
        audio_cache[original_audio_id] = {
            "data": wav_buffer.getvalue(),
            "first_accessed_at": None,
            "created_at": time.time(),
        }
        wav_buffer.seek(0)

        targetwords = demater.get_target_word_list_or_default(session_id=session_id)

        loop = asyncio.get_running_loop()
        status_queue = asyncio.Queue()

        def progress_cb(current: int, total: int):
            if total > 1:
                msg = f"⏳ Обрабатывается {current} часть из {total}..."
            else:
                msg = "⏳ Обработка аудио..."
            loop.call_soon_threadsafe(status_queue.put_nowait, msg)

        proc_task = asyncio.create_task(
            run_in_threadpool(
                demater.process,
                input_file=wav_buffer,
                target_words=targetwords,
                session_id=session_id,
                lead_in_sec=lead_in,
                lead_out_sec=lead_out,
                short_word_threshold=short_thresh,
                short_word_beep_duration=short_beep,
                max_word_beep_duration=max_beep,
                progress_callback=progress_cb,
            )
        )

        start_time = time.time()
        long_notice_sent = False

        while not proc_task.done():
            elapsed = time.time() - start_time

            if elapsed >= 15.0 and not long_notice_sent:
                long_notice_sent = True
                yield "event: progress\ndata: " + json.dumps(
                    {"status": "⚙️ Обработка продолжается (больше 15 секунд), сервер работает в штатном режиме..."},
                    ensure_ascii=False,
                ) + "\n\n"

            try:
                msg = await asyncio.wait_for(status_queue.get(), timeout=0.5)
                yield f"event: progress\ndata: {json.dumps({'status': msg}, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                # Отправляем SSE keep-alive комментарий, предотвращающий таймауты прокси
                yield ": keepalive\n\n"

        while not status_queue.empty():
            msg = status_queue.get_nowait()
            yield f"event: progress\ndata: {json.dumps({'status': msg}, ensure_ascii=False)}\n\n"

        try:
            result = await proc_task
        except Exception as e:
            logger.exception("Ошибка внутри demater.process")
            err_msg = str(e) if str(e).strip() else f"внутренняя ошибка ({type(e).__name__})"
            yield f"event: final\ndata: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
            return

        processed_audio_id = str(uuid.uuid4())
        audio_cache[processed_audio_id] = {
            "data": result["out_file"].getvalue(),
            "first_accessed_at": None,
            "created_at": time.time(),
        }

        blocks = []
        for _, method_info in result.get("methods", {}).items():
            name = method_info["name"]
            masked = markdown_spoiler_to_html(method_info["masked_text"])
            count = method_info["count"]
            blocks.append(f"<b>Вариант ({name}):</b><br/>{masked}<br/><i>Найдено слов: {count}</i>")

        formatted_msg = "<br/><br/>".join(blocks) if blocks else "Распознавание завершено."

        final_response = {
            "type": "audio",
            "text_html": formatted_msg,
            "audio_url": f"/api/audio/{processed_audio_id}",
            "original_audio_url": f"/api/audio/{original_audio_id}",
        }
        yield f"event: final\ndata: {json.dumps(final_response, ensure_ascii=False)}\n\n"

    # Заголовки, отключающие буферизацию Nginx и браузера
    sse_headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), headers=sse_headers, media_type="text/event-stream")


@app.post("/api/message")
async def handle_message(
    session_id: str = Form(...),
    text: str = Form(None),
    audio_file: UploadFile = File(None),
    lead_in: Optional[float] = Form(None),
    lead_out: Optional[float] = Form(None),
    short_thresh: Optional[float] = Form(None),
    short_beep: Optional[float] = Form(None),
    max_beep: Optional[float] = Form(None),
):
    cleanup_audio_cache()

    if audio_file:
        content = await audio_file.read()
        try:
            wav_buffer = await run_in_threadpool(convert_audio_to_wav, content)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Ошибка обработки медиафайла: {str(e)}"})

        original_audio_id = str(uuid.uuid4())
        audio_cache[original_audio_id] = {
            "data": wav_buffer.getvalue(),
            "first_accessed_at": None,
            "created_at": time.time(),
        }
        wav_buffer.seek(0)

        targetwords = demater.get_target_word_list_or_default(session_id=session_id)

        try:
            result = await run_in_threadpool(
                demater.process,
                input_file=wav_buffer,
                target_words=targetwords,
                session_id=session_id,
                lead_in_sec=lead_in,
                lead_out_sec=lead_out,
                short_word_threshold=short_thresh,
                short_word_beep_duration=short_beep,
                max_word_beep_duration=max_beep,
            )
        except Exception as e:
            logger.exception("Ошибка внутри demater.process в синхронном режиме")
            err_msg = str(e) if str(e).strip() else f"внутренняя ошибка ({type(e).__name__})"
            return JSONResponse(status_code=500, content={"error": err_msg})

        processed_audio_id = str(uuid.uuid4())
        audio_cache[processed_audio_id] = {
            "data": result["out_file"].getvalue(),
            "first_accessed_at": None,
            "created_at": time.time(),
        }

        blocks = []
        for _, method_info in result.get("methods", {}).items():
            name = method_info["name"]
            masked = markdown_spoiler_to_html(method_info["masked_text"])
            count = method_info["count"]
            blocks.append(f"<b>Вариант ({name}):</b><br/>{masked}<br/><i>Найдено слов: {count}</i>")

        formatted_msg = "<br/><br/>".join(blocks) if blocks else "Распознавание завершено."

        return {
            "type": "audio",
            "text_html": formatted_msg,
            "audio_url": f"/api/audio/{processed_audio_id}",
            "original_audio_url": f"/api/audio/{original_audio_id}",
        }

    return build_text_command_response(text, session_id)


@app.get("/api/audio/{audio_id}")
async def get_audio(audio_id: str, request: Request):
    cleanup_audio_cache()

    item = audio_cache.get(audio_id)
    if item is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Audio not found or expired"},
        )

    if item["first_accessed_at"] is None:
        item["first_accessed_at"] = time.time()

    audio_data = item["data"]
    file_size = len(audio_data)
    range_header = request.headers.get("range")

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"inline; filename={audio_id}.wav",
    }

    if range_header:
        try:
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