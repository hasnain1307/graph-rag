from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_db: str = Field(..., alias="MONGO_DB_NAME")
    database_host: str = Field(..., alias="MONGO_DATABASE_HOST")
    database_username: str = Field(..., alias="MONGO_INITDB_ROOT_USERNAME")
    database_password: str = Field(..., alias="MONGO_INITDB_ROOT_PASSWORD")
    database_port: int = Field(27017, alias="MONGO_DATABASE_PORT")

    neo4j_bolt_uri: str = Field(..., alias="NEO4J_BOLT_URI")
    neo4j_user: str = Field(..., alias="NEO4J_USER")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    pgvector_host: str = Field(..., alias="PGVECTOR_HOST")  # Add this to .env
    pgvector_port: int = Field(..., alias="PGVECTOR_PORT")
    pgvector_user: str = Field(..., alias="PGVECTOR_USER")
    pgvector_password: str = Field(..., alias="PGVECTOR_PASSWORD")
    pgvector_db: str = Field(..., alias="PGVECTOR_DB")

    @property
    def _database_url(self):
        return f"{self.database_host}:{self.database_port}"

    @property
    def neo4j_auth(self):
        auth = (self.neo4j_user, self.neo4j_password)
        return auth

    @property
    def mongo_database_conn_str(self) -> str:
        return f"mongodb://{self.database_username}:{self.database_password}@{self._database_url}/{self.database_db}?authSource=admin"

    @property
    def pgvector_async_url(self) -> str:
        """PostgreSQL async connection URL for pgvector (asyncpg)"""
        return f"postgresql+asyncpg://{self.pgvector_user}:{self.pgvector_password}@{self.pgvector_host}:{self.pgvector_port}/{self.pgvector_db}"

config = Settings()
