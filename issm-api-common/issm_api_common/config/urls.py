from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_db: str = Field(..., alias="MONGO_DB_NAME")
    database_host: str = Field(..., alias="MONGO_DATABASE_HOST")
    database_username: str = Field(..., alias="MONGO_INITDB_ROOT_USERNAME")
    database_password: str = Field(..., alias="MONGO_INITDB_ROOT_PASSWORD")
    database_port: int = Field(27017, alias="MONGO_DATABASE_PORT")

    @property
    def _database_url(self):
        return f"{self.database_host}:{self.database_port}"

    @property
    def mongo_database_conn_str(self) -> str:
        return f"mongodb://{self.database_username}:{self.database_password}@{self._database_url}/{self.database_db}?authSource=admin"


config = Settings()
