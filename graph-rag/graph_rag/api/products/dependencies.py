from llm_as_judge.api.products.services import ProductService
from llm_as_judge import injector


def get_product_service() -> ProductService:
    """Returns an ProductService object"""
    return injector.get(ProductService)
