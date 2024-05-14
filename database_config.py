from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    candlesticks_schema: str
    database_username: str
    database_read_only_username: str
    database_read_only_password: str

    class Config:
        env_file = Path(__file__).parent.joinpath('.env')


settings = Settings()
