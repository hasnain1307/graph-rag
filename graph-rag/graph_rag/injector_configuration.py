from injector import singleton, provider, inject, Module
from neo4j import GraphDatabase, Driver
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from issm_api_common.config.urls import config


class InjectorConfiguration(Module):
    @singleton
    @provider
    @inject
    def provide_neo4j_client(self) -> Driver:
        return GraphDatabase.driver(config.neo4j_bolt_uri, auth=config.neo4j_auth)

    @singleton
    @provider
    @inject
    def provide_pgvector_client(self) -> AsyncEngine:
        engine = create_async_engine(
            url=config.pgvector_async_url,
            pool_size=20,
            max_overflow=10,
            echo=False
        )
        return engine
