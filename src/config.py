import os
import yaml
from pydantic import BaseModel
from typing import Optional


class RetrieverConfig(BaseModel):
    conversations_paths: list[str]
    embedding_model: str
    device: str
    k: int


class Config(BaseModel):
    debug: bool = False
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    ollama_host: str = "ollama"
    ollama_port: int = 11435
    tg_app_id: int = None
    tg_app_hash: str = None
    tg_bot_token: str = None
    system_prompt_path: str = None
    retriever: Optional[RetrieverConfig] = None

    @classmethod
    def from_yaml(cls, path: str):
        config = yaml.safe_load(open(path))
        config["ollama_host"] = os.environ.get("OLLAMA_HOST", config.get("ollama_host", "ollama"))
        config["ollama_port"] = os.environ.get("OLLAMA_PORT", config.get("ollama_port", 11435))
        config["redis_host"] = os.environ.get("REDIS_HOST", config.get("redis_host", "redis"))
        config["redis_port"] = os.environ.get("REDIS_PORT", config.get("redis_port", 6379))
        config["redis_db"] = os.environ.get("REDIS_DB", config.get("redis_db", 0))
        return cls(**config, strict=True)


def load_config(path: str = "config.yaml"):
    return Config.from_yaml(os.environ.get("LITTLE_ANTIPOV_CONFIG_PATH", path))


