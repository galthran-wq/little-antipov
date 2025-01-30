import mimetypes
import io
import os
import sys
from tempfile import NamedTemporaryFile
import logging
from dataclasses import dataclass, asdict
import json
from urllib.parse import urlparse
from collections import defaultdict

import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, ConversationHandler, MessageHandler
from telegram.ext import filters
import re

from src.config import load_config
config = load_config()
token = config.tg_bot_token

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

SYSTEM = USERNAME = ID = CHAT = range(1)
user_id_to_dialog_id = {}
user_id_to_system = {}

def prepare_message_for_antipov_bot(id, message: str):
    return f"from {id}: {message}"

def prep_text(s):
    import re
    # return text.replace(".", "\.").replace("(", "\(").replace(")", "\)").replace("-", "\-").replace("+", "\+").replace("*", "\*")
    s = (re.sub(r'_', r'\\_', s)
        .replace("-", r"\-")
        .replace("~", r"\~")
        .replace("`", r"\`")
        .replace(".", r"\.")
        .replace("+", r"\+")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("*", r"\*"))
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    s = emoji_pattern.sub('', s)
    return s

async def send_text(context: ContextTypes.DEFAULT_TYPE, text, chat_id):
    current_chunk = []
    current_len = 0
    for line in text.split("\n"):
        if current_len + len(line) > 4_000:
            await context.bot.send_message(chat_id=chat_id, text="\n".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += len(line)
    if len(current_chunk) > 0:
        await context.bot.send_message(chat_id=chat_id, text="\n".join(current_chunk))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = f"Я ассистент. Чтобы закончить разговор введи /stop"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    return CHAT

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_id_to_dialog_id
    id = update.effective_user.id
    image = None
    document = None
    file_name = None
    text = update.effective_message.text if update.effective_message.text else ""

    # Check if a photo is uploaded
    if len(update.message.photo) > 0:
        image = io.BytesIO()
        photo_file = await update.message.photo[-1].get_file()
        file_name = photo_file.file_path
        await photo_file.download_to_memory(image)
        image.seek(0)
        text = update.message.caption  # Add caption if available

    # Check if a document (file) is uploaded
    elif update.message.document:
        document = io.BytesIO()
        doc_file = await update.message.document.get_file()
        file_name = doc_file.file_path
        await doc_file.download_to_memory(document)
        document.seek(0)
        text = update.message.caption  # Add caption if available
    
    if id not in user_id_to_dialog_id:
        user_id_to_dialog_id[id] = abs(id)

    # Prepare the request data
    files = {
        'message': (None, json.dumps({
            "model": "little-antipov-llama-3.1-7b:latest",
            "text": prepare_message_for_antipov_bot(id, text), 
            "thread_id": str(user_id_to_dialog_id[id]),
            "system": user_id_to_system.get(id, None),
        })),
    }

    # Add the image or document to the request if they exist
    if image is not None:
        files['image'] = (file_name, image, mimetypes.guess_type(file_name)[0])
    elif document is not None:
        files['doc'] = (file_name, document, mimetypes.guess_type(file_name)[0])

    # Send the request to the API
    result = requests.post(
        url="http://127.0.0.1:8000/message",
        files=files,
    ).content.decode()

    try:
        # Process the response
        result = json.loads(result)['text']
        await send_text(
            context=context,
            chat_id=update.effective_chat.id,
            text=result,
        )
    except json.decoder.JSONDecodeError:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Ошибка!\n{result}",
        )
    except KeyError:
        print(result)

    return CHAT

        
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="пока :(")
    return -1

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_id_to_dialog_id
    id = update.effective_user.id
    if id in user_id_to_dialog_id:
        import random
        user_id_to_dialog_id[id] += random.randint(0, 100)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="готово")

async def system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="введи новый system prompt (или /stop)")
    return SYSTEM

async def set_system(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_id_to_system
    id = update.effective_user.id
    user_id_to_system[id] = update.effective_message.text
    await context.bot.send_message(chat_id=update.effective_chat.id, text="готово")
    return -1


if __name__ == '__main__':
    # Enable logging
    help_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHAT: [MessageHandler((filters.TEXT | filters.PHOTO | filters.ATTACHMENT) & ~filters.COMMAND, chat)],
        },
        fallbacks=[CommandHandler("stop", cancel)],
    )
    system_handler = ConversationHandler(
        entry_points=[CommandHandler("system", system)],
        states={
            SYSTEM: [MessageHandler((filters.TEXT) & ~filters.COMMAND, set_system)],
        },
        fallbacks=[CommandHandler("stop", cancel)],
    )
    reset_handler = CommandHandler("reset", reset)
    # set higher logging level for httpx to avoid all GET and POST requests being logged
    logging.getLogger("httpx").setLevel(logging.WARNING)
    application = ApplicationBuilder().token(token).build()
    application.add_handler(help_handler)
    application.add_handler(system_handler)
    application.add_handler(reset_handler)
    application.run_polling()
