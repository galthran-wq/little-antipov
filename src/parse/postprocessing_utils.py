from tqdm.auto import tqdm
from datetime import datetime, timedelta


def is_valid_conversation(conversation):
    """
    Check if a conversation is valid.

    A valid conversation must have more than one message and include at least one message
    from both "gpt" and "human".

    Args:
        conversation (list): A list of message dictionaries, where each dictionary contains
                             a "from" key indicating the sender ("gpt" or "human").

    Returns:
        bool: True if the conversation is valid, False otherwise.
    """
    return (
        len(conversation) > 1 and
        any(message["from"] == "gpt" for message in conversation) and
        any(message["from"] == "human" for message in conversation)
    )

def extract_conversations_from_personal_chat(
    df, 
    target_author_id=0, 
    hours_delta=6
):
    """
    Extract conversations from a personal chat DataFrame.

    This function processes a DataFrame containing chat messages and extracts conversations
    based on a specified time delta. Conversations are grouped by messages exchanged within
    a certain number of hours.

    Args:
        df (pandas.DataFrame): DataFrame containing chat messages with columns "Author ID",
                               "Content", and "Date".
        target_author_id (int, optional): The ID of the target author (e.g., "gpt"). Defaults to 0.
        hours_delta (int, optional): The maximum number of hours between messages to consider them
                                     part of the same conversation. Defaults to 6.

    Returns:
        list: A list of valid conversations, where each conversation is a list of message dictionaries.
    """
    participants_ids = df["Author ID"].unique().tolist()
    assert target_author_id in participants_ids
    other_participant_id = [
        participant_id
        for participant_id in participants_ids
        if participant_id != target_author_id
    ][0]
    conversations = []
    current_conversation = []
    previous_message_date = datetime.strptime(df.iloc[-1]["Date"], '%Y-%m-%d %H:%M:%S')
    
    for i, row in tqdm(df.iloc[-2::-1].iterrows()):
        message_author_id = row["Author ID"]
        message_content = row["Content"]
        if not message_content:
            continue
        # example date: '2025-01-28 18:05:54'
        message_date = datetime.strptime(row["Date"], '%Y-%m-%d %H:%M:%S')
        from_ = "gpt" if message_author_id == target_author_id else "human"
        message = {
            "from": from_,
            "value": (
                f"from {other_participant_id} to {target_author_id}: {message_content}"
                if from_ == "human"
                else 
                f"to {other_participant_id}: {message_content}"
            )
        }
        # if previous message is within hours_delta hours, then it is the same conversation
        if message_date - previous_message_date < timedelta(hours=hours_delta):
            current_conversation.append(message)
        else:
            conversations.append(current_conversation)
            current_conversation = [message]
        previous_message_date = message_date
    
    print(f"Found {len(conversations)} conversations")
    valid_conversations = [conversation for conversation in conversations if is_valid_conversation(conversation)]
    print(f"Found {len(valid_conversations)} valid conversations")
    return valid_conversations


def prepare_conversations(conversations):
    """
    Prepare conversations for further processing.

    This function processes a list of conversations and prepares them for further processing.
    It combines consecutive messages from the same author into a single message.

    Args:
        conversations (list): A list of conversations, where each conversation is a list of message dictionaries.

    Returns:
        list: A list of prepared conversations, where each conversation is a list of message dictionaries.
    """
    prepared_conversations = []
    for conversation in conversations:
        if not conversation:
            continue
        prepared_conversation = []
        current_message = conversation[0]["value"]
        current_from = conversation[0]["from"]
        
        for message in conversation[1:]:
            if message["from"] == current_from:
                current_message += "\n" + message["value"]
            else:
                prepared_conversation.append({"from": current_from, "value": current_message})
                current_message = message["value"]
                current_from = message["from"]
        
        # Append the last accumulated message
        prepared_conversation.append({"from": current_from, "value": current_message})
        prepared_conversations.append(prepared_conversation)
    
    return prepared_conversations