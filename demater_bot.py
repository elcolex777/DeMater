#!/usr/bin/env python
# pylint: disable=unused-argument
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Application and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

from telegram import ForceReply, ReplyKeyboardRemove, Update
from telegram.ext import Application, CommandHandler, ContextTypes, ConversationHandler, MessageHandler, filters

import os
import io
from demater import DeMater
import soundfile as sf

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


model_path = os.environ.get('DEMATBOT_MODEL_PATH') if os.environ.get('DEMATBOT_MODEL_PATH') is not None else "models/vosk-model-small-ru-0.22"
demater = DeMater(model_path=model_path)
#demater = DeMater(model_path="models/vosk-model-ru-0.42")
#demater = DeMater(model_path="models/vosk-model-small-en-us-0.15")
#demater = DeMater()


TARGETWORDS_SET = 1
TARGETWORDS_ADD = 2

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"""Привет!
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
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords.split(",")

    
    targetwords = demater.replace_text(update.message.text, targetwords)
    await update.message.reply_text(targetwords , parse_mode='MarkdownV2')

    #await update.message.reply_text(update.message.text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "OK", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END

async def target_words(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    
    targetwords_new = update.message.text.replace("/targetwords", "")
    if len(targetwords_new) > 0:
        demater.get_user_data_or_new(update.message.chat.id)["target_word_list_custom"] = targetwords_new.strip().replace(" ", ",").lower()

    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords.split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked = targetwords_masked + ("\.\.\." if len(targetwords) >= 20 else "")
    await update.message.reply_text(targetwords_masked , parse_mode='MarkdownV2')

async def targetwords_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    
    await update.message.reply_text(
         "Использовать свой список \"матерных\" слов\. \nИли /cancel для отмены ввода"
         "\n\nСписок слов через пробел\:", 
         parse_mode='MarkdownV2'
    )

    return TARGETWORDS_SET


async def targetwords_set_end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    
    targetwords_new = update.message.text
    if len(targetwords_new) > 0:
        demater.get_user_data_or_new(update.message.chat.id)["target_word_list_custom"] = targetwords_new.strip().replace(" ", ",").lower()

    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords.split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked = targetwords_masked + ("\.\.\." if len(targetwords) >= 20 else "")
    await update.message.reply_text(targetwords_masked , parse_mode='MarkdownV2')

async def targetwords_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    
    await update.message.reply_text(
         "Добавить свой список \"матерных\" слов к текущему\. \nИли /cancel для отмены ввода"
         "\n\nСписок слов через пробел\:", 
         parse_mode='MarkdownV2'
    )

    return TARGETWORDS_ADD

async def targetwords_add__end(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    
    targetwords_new = update.message.text
    if len(targetwords_new) > 0:
        demater.get_user_data_or_new(update.message.chat.id)["target_word_list_custom"] = targetwords_new.strip().replace(" ", ",").lower()\
            + ',' + demater.get_target_word_list_or_default(session_id=update.message.chat.id)

    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords.split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked = targetwords_masked + ("\.\.\." if len(targetwords) >= 20 else "")
    await update.message.reply_text(targetwords_masked , parse_mode='MarkdownV2')

    return ConversationHandler.END

async def targetwords_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    
    demater.user_data[update.message.chat.id]["target_word_list_custom"] = ""

    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords.split(",")[:20]
    targetwords_masked = demater.replace_text(" ".join(targetwords), targetwords)
    targetwords_masked = targetwords_masked + ("\.\.\." if len(targetwords) >= 20 else "")
    await update.message.reply_text(targetwords_masked , parse_mode='MarkdownV2')

async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print('call audio start')
    #demater = DeMater()
    buffer = io.BytesIO()
    new_file = await context.bot.get_file(update.message.voice.file_id)
    await new_file.download_to_memory(out=buffer)

    buffer.seek(0)
    data, samplerate = sf.read(buffer)

    buffer = io.BytesIO()
    sf.write(file=buffer, data=data, samplerate=samplerate, format='WAV')

    await update.message.reply_text("начинаем обработку\.\.\." , parse_mode='MarkdownV2')

    buffer.seek(0)
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    result = demater.process(input_file=buffer, target_words=targetwords, session_id=update.message.chat.id)

    await update.message.reply_text(
        rf"""Вариант 1:
{result["text"]}
Количество матерных слов: {result["detected_word_list_count"]}""" , parse_mode='MarkdownV2')
    await update.message.reply_text(
        rf"""Вариант 2:
{result["text_whisper"]}
Количество матерных слов: {result["detected_word_list2_count"]}""" , parse_mode='MarkdownV2')

    await update.message.reply_audio(audio=result["out_file"], message_effect_id="5046509860389126442", filename="audio.wav")

async def document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print('call audio start')
    #demater = DeMater()
    buffer = io.BytesIO()
    new_file = await context.bot.get_file(update.message.document.file_id)
    await new_file.download_to_memory(out=buffer)
    
    await update.message.reply_text("начинаем обработку\.\.\." , parse_mode='MarkdownV2')

    buffer.seek(0)
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    result = demater.process(input_file=buffer, target_words=targetwords, session_id=update.message.chat.id)

    await update.message.reply_text(
        rf"""Вариант 1:
{result["text"]}
Количество матерных слов: {result["detected_word_list_count"]}""" , parse_mode='MarkdownV2')
    await update.message.reply_text(
        rf"""Вариант 2:
{result["text_whisper"]}
Количество матерных слов: {result["detected_word_list2_count"]}""" , parse_mode='MarkdownV2')

    await update.message.reply_audio(audio=result["out_file"], message_effect_id="5046509860389126442", filename="audio.wav")

async def audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print('call audio start')
    #demater = DeMater()
    buffer = io.BytesIO()
    new_file = await context.bot.get_file(update.message.audio.file_id)
    await new_file.download_to_memory(out=buffer)

    buffer.seek(0)
    data, samplerate = sf.read(buffer)

    buffer = io.BytesIO()
    sf.write(file=buffer, data=data, samplerate=samplerate, format='WAV')

    await update.message.reply_text("начинаем обработку\.\.\." , parse_mode='MarkdownV2')

    buffer.seek(0)
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    result = demater.process(input_file=buffer, target_words=targetwords, session_id=update.message.chat.id)

    await update.message.reply_text(
        rf"""Вариант 1:
{result["text"]}
Количество матерных слов: {result["detected_word_list_count"]}""" , parse_mode='MarkdownV2')
    await update.message.reply_text(
        rf"""Вариант 2:
{result["text_whisper"]}
Количество матерных слов: {result["detected_word_list2_count"]}""" , parse_mode='MarkdownV2')

    await update.message.reply_audio(audio=result["out_file"], message_effect_id="5046509860389126442", filename="audio.wav")


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    token = os.environ.get('DEMATBOT_TOKEN')
    application = Application.builder().token(token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.VOICE, voice))
    application.add_handler(MessageHandler(filters.AUDIO, audio))
    application.add_handler(MessageHandler(filters.ATTACHMENT, document))
    print('handler audio register')

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

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
