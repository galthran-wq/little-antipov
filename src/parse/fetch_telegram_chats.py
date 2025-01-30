import json
from typing import Optional
import re
import time

import asyncio
import fire
import pandas as pd
from tqdm.auto import tqdm
from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetDiscussionMessageRequest
from telethon.tl.functions.channels import GetMessagesRequest
from telethon.tl.types import InputChannel, InputPeerChannel

from src.config import load_config


def setup_telegram_client() -> TelegramClient:
    config = load_config()

    # Use your own values from my.telegram.org
    api_id = config.tg_app_id
    api_hash = config.tg_app_hash

    # Create the client and connect
    client = TelegramClient('session_name', api_id, api_hash)
    return client


def remove_unsupported_characters(text):
    valid_xml_chars = (
        "[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD"
        "\U00010000-\U0010FFFF]"
    )
    try:
        cleaned_text = re.sub(valid_xml_chars, '', text)
        return cleaned_text
    except Exception as e:
        print(f"Error removing unsupported characters: {e}")
        print(f"Text: {text}")
        return text

# Function to format time in days, hours, minutes, and seconds
def format_time(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f'{int(days):02}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}'


async def main(
    channel: str, 
    output_file: str, 
    limit: int = 99999999, 
    min_id: int = 0, 
    max_id: int = 0,
    ids: list[int] | int = None
):
    data = []

    # Tracker for the number of messages processed
    t_index = 0
    c_index = 0
    try:
        async with setup_telegram_client() as client:
            client: TelegramClient
            async for message in tqdm(
                client.iter_messages(
                    channel, 
                    limit=limit, 
                    min_id=min_id, 
                    max_id=max_id, 
                    ids=ids
                ), 
                desc=f"Processing {channel} posts (limit={limit})..."
            ):
                # Process the main message
                media = 'True' if message.media else 'False'

                emoji_string = ''
                if message.reactions:
                    for reaction_count in message.reactions.results:
                        if hasattr(reaction_count.reaction, 'emoticon'):
                            emoji = reaction_count.reaction.emoticon
                            count = str(reaction_count.count)
                            emoji_string += emoji + " " + count + " "

                date_time = message.date.strftime('%Y-%m-%d %H:%M:%S')
                cleaned_content = remove_unsupported_characters(message.text)

                data.append({
                    'Type': 'text',
                    'Group': channel,
                    'Author ID': message.sender_id,
                    'Content': cleaned_content,
                    'Date': date_time,
                    'Message ID': message.id,
                    'Author': message.post_author,
                    'Views': message.views,
                    'Reactions': emoji_string,
                    'Shares': message.forwards,
                    'Media': media,
                })

                c_index += 1
                t_index += 1

                # Print progress
                print(f'{"-" * 80}')
                print(f'Id: {message.id:05} / Date: {date_time}')
                print(f'Total: {t_index:05} contents until now')
                print(f'{"-" * 80}\n\n')

                if t_index % 1000 == 0:
                    backup_filename = f'backup_{output_file}_until_{t_index:05}_{channel}_ID{message.id:07}.parquet'
                    pd.DataFrame(data).to_parquet(backup_filename, index=False)

        print(f'\n\n##### {channel} was ok with {c_index:05} posts #####\n\n')

        df = pd.DataFrame(data)
        partial_filename = f'complete_{channel}_in_{output_file}_until_{t_index:05}.parquet'
        df.to_parquet(partial_filename, index=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'{channel} error: {e}')


async def run_main(
    channel: str,
    output_file: str
):
    await main(channel, output_file)


if __name__ == "__main__":
    fire.Fire(run_main)
