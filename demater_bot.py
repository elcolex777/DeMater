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

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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

Использовать свой список "матерных" слов
/targetwords список,слов,через,запятую

Сбросить свой список "матерных" слов
/targetwords_reset


        """,
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)

async def target_words(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    
    targetwords_new = update.message.text.replace("/targetwords", "")
    if len(targetwords_new) > 0:
        demater.target_word_list_custom[update.message.chat.id] = targetwords_new.strip()

    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords[:100] + ("\.\.\." if len(targetwords) > 100 else "")
    targetwords = demater.mask_text(targetwords, targetwords.split(" "))
    await update.message.reply_text(targetwords , parse_mode='MarkdownV2')

async def targetwords_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    
    demater.target_word_list_custom[update.message.chat.id] = ""

    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    targetwords = targetwords[:100] + ("\.\.\." if len(targetwords) > 100 else "")
    targetwords = demater.mask_text(targetwords, targetwords.split(" "))
    await update.message.reply_text(targetwords , parse_mode='MarkdownV2')

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

    buffer.seek(0)
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    result = demater.process(input_file=buffer, target_words=targetwords)

    await update.message.reply_text(
        rf"""{result["text"]}
Количество матерных слов: {result["detected_word_list_count"]}""" , parse_mode='MarkdownV2')

    await update.message.reply_audio(audio=result["out_file"], message_effect_id="5046509860389126442", filename="audio.wav")

async def document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print('call audio start')
    #demater = DeMater()
    buffer = io.BytesIO()
    new_file = await context.bot.get_file(update.message.document.file_id)
    await new_file.download_to_memory(out=buffer)

    buffer.seek(0)
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    result = demater.process(input_file=buffer, target_words=targetwords)
    print(f'call audio end: {result["text"]}')

    await update.message.reply_text(
        rf"""{result["text"]}
Количество матерных слов: {result["detected_word_list_count"]}""" , parse_mode='MarkdownV2')

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

    buffer.seek(0)
    targetwords = demater.get_target_word_list_or_default(session_id=update.message.chat.id)
    result = demater.process(input_file=buffer, target_words=targetwords)

    await update.message.reply_text(
        rf"""{result["text"]}
Матерных слов: {result["detected_word_list_count"]}""" , parse_mode='MarkdownV2')

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

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
