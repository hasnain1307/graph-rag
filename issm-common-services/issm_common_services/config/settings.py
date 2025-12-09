from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_to_file: bool = Field(True, env="LOG_TO_FILE")
    log_to_console: bool = Field(True, env="LOG_TO_CONSOLE")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file_path: str = Field("logs", env="LOG_FILE_PATH")


config = Settings()
