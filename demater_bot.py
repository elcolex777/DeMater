#!/usr/bin/env python
import asyncio
import io
import logging
import os

import soundfile as sf
from telegram import ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from demater import DeMater

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

model_path = os.environ.get("DEMATBOT_MODEL_PATH", "models/vosk-model-small-ru-0.22")

# Конфигурация моделей: Vosk и Whisper Fast включены, старый Whisper отключен
demater = DeMater(
    vosk_model_path=model_path,
    enable_vosk=True,
    enable_faster_whisper=True,
    enable_whisper=False,
)

TARGETWORDS_SET = 1
TARGETWORDS_ADD = 2


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(
        """Привет!
Этот бот запикивает части аудио с матом. 
Просто отправьте голосовое в чат или приложите аудиофайл.

Посмотреть текущий список "матерных" слов:
/targetwords

Использовать свой список "матерных" слов:
/targetwords_set
список слов через пробел

Добавить свой список "матерных" слов к основному:
/targetwords_add
список слов через пробел

Сбросить свой список "матерных" слов:
/targetwords_reset
"""
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Отправьте голосовое сообщение или аудиофайл для обработки.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    targetwords = demater.get_target_word_list_or_default(session_id=str(update.message.chat.id))
    targetwords = targetwords.split(",")
    result_text = demater.replace_text(update.message.text, targetwords)
    await update.message.reply_text(result_text, parse_mode="MarkdownV2")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("OK", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def target_words(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    targetwords_new = update.message.text.replace("/targetwords", "")
    chat_id = str(update.message.chat.id)
    if len(targetwords_new) > 0:
        demater.get_user_data_or_new(chat_id)["target_word_list_custom"] = targetwords_new.strip().replace(" ", ",").lower()

    targetwords = demater.get_target_word_list_or_default(session_id=chat_id).split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked += r"\.\.\." if len(targetwords) >= 20 else ""
    await update.message.reply_text(targetwords_masked, parse_mode="MarkdownV2")


async def targetwords_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Использовать свой список \"матерных\" слов\\. \nИли /cancel для отмены ввода"
        "\n\nСписок слов через пробел\\:",
        parse_mode="MarkdownV2",
    )
    return TARGETWORDS_SET


async def targetwords_set_end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = str(update.message.chat.id)
    targetwords_new = update.message.text
    if len(targetwords_new) > 0:
        demater.get_user_data_or_new(chat_id)["target_word_list_custom"] = targetwords_new.strip().replace(" ", ",").lower()

    targetwords = demater.get_target_word_list_or_default(session_id=chat_id).split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked += r"\.\.\." if len(targetwords) >= 20 else ""
    await update.message.reply_text(targetwords_masked, parse_mode="MarkdownV2")
    return ConversationHandler.END


async def targetwords_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Добавить свой список \"матерных\" слов к текущему\\. \nИли /cancel для отмены ввода"
        "\n\nСписок слов через пробел\\:",
        parse_mode="MarkdownV2",
    )
    return TARGETWORDS_ADD


async def targetwords_add__end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = str(update.message.chat.id)
    targetwords_new = update.message.text
    if len(targetwords_new) > 0:
        demater.get_user_data_or_new(chat_id)["target_word_list_custom"] = (
            targetwords_new.strip().replace(" ", ",").lower()
            + ","
            + demater.get_target_word_list_or_default(session_id=chat_id)
        )

    targetwords = demater.get_target_word_list_or_default(session_id=chat_id).split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked += r"\.\.\." if len(targetwords) >= 20 else ""
    await update.message.reply_text(targetwords_masked, parse_mode="MarkdownV2")
    return ConversationHandler.END


async def targetwords_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.message.chat.id)
    if chat_id in demater.user_data:
        demater.user_data[chat_id]["target_word_list_custom"] = ""

    targetwords = demater.get_target_word_list_or_default(session_id=chat_id).split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked += r"\.\.\." if len(targetwords) >= 20 else ""
    await update.message.reply_text(targetwords_masked, parse_mode="MarkdownV2")


async def _handle_audio_common(update: Update, context: ContextTypes.DEFAULT_TYPE, file_obj) -> None:
    buffer = io.BytesIO()
    new_file = await context.bot.get_file(file_obj.file_id)
    await new_file.download_to_memory(out=buffer)
    buffer.seek(0)

    # Приводим к PCM 16kHz mono через soundfile
    data, samplerate = sf.read(buffer)
    wav_buffer = io.BytesIO()
    sf.write(file=wav_buffer, data=data, samplerate=samplerate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)

    await update.message.reply_text("Начинаем обработку\\.\\.\\.", parse_mode="MarkdownV2")

    chat_id = str(update.message.chat.id)
    targetwords = demater.get_target_word_list_or_default(session_id=chat_id)

    # Асинхронный вызов в пуле потоков
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, demater.process, wav_buffer, targetwords, chat_id
    )

    # Динамически выводим результаты только тех методов, которые были включены
    for _, method_info in result.get("methods", {}).items():
        name = method_info["name"]
        text = method_info["masked_text"]
        count = method_info["count"]
        msg = f"*{name}*\n{text}\nКоличество матерных слов: {count}"
        await update.message.reply_text(msg, parse_mode="MarkdownV2")

    await update.message.reply_audio(
        audio=result["out_file"],
        filename="censored.wav",
    )


async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_audio_common(update, context, update.message.voice)


async def document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_audio_common(update, context, update.message.document)


async def audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _handle_audio_common(update, context, update.message.audio)


def main() -> None:
    token = os.environ.get("DEMATBOT_TOKEN")
    if not token:
        raise ValueError("DEMATBOT_TOKEN environment variable not set")

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.VOICE, voice))
    application.add_handler(MessageHandler(filters.AUDIO, audio))
    application.add_handler(MessageHandler(filters.ATTACHMENT, document))

    application.add_handler(CommandHandler("targetwords", target_words))
    application.add_handler(CommandHandler("targetwords_reset", targetwords_reset))

    conv_handler_add = ConversationHandler(
        entry_points=[CommandHandler("targetwords_add", targetwords_add)],
        states={
            TARGETWORDS_ADD: [MessageHandler(filters.TEXT & ~filters.COMMAND, targetwords_add__end)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(conv_handler_add)

    conv_handler_set = ConversationHandler(
        entry_points=[CommandHandler("targetwords_set", targetwords_set)],
        states={
            TARGETWORDS_SET: [MessageHandler(filters.TEXT & ~filters.COMMAND, targetwords_set_end)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(conv_handler_set)

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()