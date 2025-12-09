from graph_rag.api.products.services import ProductService
from graph_rag import injector


def get_product_service() -> ProductService:
    """Returns an ProductService object"""
    return injector.get(ProductService)
