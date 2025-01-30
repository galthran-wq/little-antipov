# Little Antipov Chatbot

This project provides an end-to-end solution for creating a custom chatbot. It includes utilities for collecting data from Telegram, training a model, and deploying a containerized REST API with a Telegram bot interface.

## Key Features

1. **Chat-like Design**: The `/message` endpoint accepts a dialogue ID and automatically remembers the context of the dialogue.
2. **Message History Management**: Capable of truncating message history if it exceeds the context length.
3. **Efficient Memory Management**: Keeps chat history in memory and uses Redis for speed-efficient storage.
4. **Model Serving**: Utilizes Ollama to serve local (self-trained and other) models.

## Example Usage

To interact with the chatbot via the REST API, you can use the following `curl` command:

```bash
curl -X POST http://localhost:8000/message \
     -F 'message={"model": "your-model-name", "text": "Hello, how are you?", "thread_id": "12345"}'
```

## Repository Structure

- `src/parse/fetch_telegram_chats.py`: Utilities for parsing Telegram chats.
- `src/train/sft.py`: Script for fine-tuning the model using SFT (Supervised Fine-Tuning).
- `src/bot/telegram_bot.py`: Telegram bot interface for interacting with the chatbot.
- `src/agent.py`: Core logic for managing chat sessions and interactions.
- `src/redis_chain_memory.py`: Implementation of Redis-based checkpoint saver for efficient memory management.
- `docker-compose.yaml`: Configuration for containerized deployment using Docker.
- `config.py`: Configuration management using Pydantic and YAML.

## Creating Your Own Chatbot

1. **Collect Telegram Data**: Use the provided utilities to collect data from personal Telegram conversations and public group comments to mimic character behavior.
2. **Post-process Data**: Convert the collected data into the required conversation format. Example:
   ```json
   [{"from": "human", "value": "from 392065318: привет как дела"}, {"from": "gpt", "value": "to 392065318: хорошо"}]
   ```
3. **Fine-tune the Model**: Use the SFT trainer script to fine-tune the model.
4. **Model Conversion (Undocumented)**: Use Ollama and llama.cpp to merge the adapter with the base model and convert it to GGUF format.
5. **Load the Model (Undocumented)**: Load the model into the local Ollama instance for serving.
