from injector import singleton, provider, inject, Module

from llm_as_judge.api.products.services import ProductService
from issm_common_database_setup.mongo import BeanieDBClient
from issm_api_common.config.urls import config


class InjectorConfiguration(Module):
    @singleton
    @provider
    @inject
    def provide_mongo_db_client(self) -> BeanieDBClient:
        return BeanieDBClient(config.mongo_database_conn_str)

    @singleton
    @provider
    @inject
    def provide_product_service(self) -> ProductService:
        return ProductService()
