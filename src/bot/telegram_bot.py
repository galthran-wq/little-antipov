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
import asyncio
import secrets

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

async def send_text(context: ContextTypes.DEFAULT_TYPE, text, chat_id, reply_to_message_id=None):
    for line in text.split("\n"):
        if line.strip():
            await context.bot.send_message(chat_id=chat_id, text=line, reply_to_message_id=reply_to_message_id)
            # Introduce a delay based on the length of the line
            delay = min(2, len(line) / 20)  # Example: 0.1 seconds per character, capped at 2 seconds
            await asyncio.sleep(delay)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = f"Я ассистент. Чтобы закончить разговор введи /stop"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    return CHAT

def generate_dialog_id():
    return secrets.token_hex(16)  # Generates a random 32-character hexadecimal string

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
        user_id_to_dialog_id[id] = generate_dialog_id()

    # Prepare the request data
    files = {
        'message': (None, json.dumps({
            "model": "llama3.1:8b-retrieve-antipov",
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
    try:
        result = requests.post(
            url="http://127.0.0.1:8000/message",
            files=files,
        ).content.decode()

        # Process the response
        result = json.loads(result)['text']
        
        # Check if the response starts with "to {id}"
        if result.startswith(f"to {id}:"):
            reply_to_message_id = update.effective_message.message_id
            result = result[len(f"to {id}:"):].strip()
        else:
            reply_to_message_id = None

        await send_text(
            context=context,
            chat_id=update.effective_chat.id,
            text=result,
            reply_to_message_id=reply_to_message_id
        )
    except (json.decoder.JSONDecodeError, KeyError) as e:
        # Reset the state
        user_id_to_dialog_id[id] = generate_dialog_id()
        # Log the error
        logging.error(f"Error occurred: {e}")
        await chat(update, context)

    return CHAT

        
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="пока :(")
    return -1

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global user_id_to_dialog_id
    id = update.effective_user.id
    user_id_to_dialog_id[id] = generate_dialog_id()
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
    chat_handler = MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.ATTACHMENT) & ~filters.COMMAND, chat
    )
    # set higher logging level for httpx to avoid all GET and POST requests being logged
    logging.getLogger("httpx").setLevel(logging.WARNING)
    application = ApplicationBuilder().token(token).build()
    application.add_handler(chat_handler)
    application.run_polling()
