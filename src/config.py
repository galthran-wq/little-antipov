import os
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    debug: bool = False
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    ollama_host: str = "ollama"
    ollama_port: int = 11435

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


