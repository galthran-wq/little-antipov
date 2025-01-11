from pydantic import BaseModel
import yaml

class Config(BaseModel):
    telegram_app_id: int
    telegram_app_hash: str
    telegram_channel_username: str

    @classmethod
    def from_yaml(cls, path: str = "config.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

