from pydantic import BaseModel


class TeleoperateRequest(BaseModel):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str
