import os
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    debug: bool = False
    redis_url: str = "redis://localhost:6379"
    redis_port: int = 6379
    redis_db: int = 0

    @classmethod
    def from_yaml(cls, path: str):
        return cls(**yaml.safe_load(open(path)), strict=True)


def load_config(path: str = "config.yaml"):
    return Config.from_yaml(os.environ.get("LITTLE_ANTIPOV_CONFIG_PATH", path))


