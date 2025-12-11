from graph_rag.api.docs_ingestion.services import GraphIngestionService
from graph_rag import injector


def get_graph_ingestion_service() -> GraphIngestionService:
    """Returns an ProductService object"""
    return injector.get(GraphIngestionService)
