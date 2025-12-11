from fastapi import APIRouter, status, Depends, HTTPException, Query
from typing import List, Dict, Any
from graph_rag.logger import logger
from graph_rag.api.docs_ingestion.models import (
    TripleIngestionRequest,
    TripleIngestionResponse,
)
from graph_rag.api.docs_ingestion.services import GraphIngestionService
from graph_rag.api.docs_ingestion.dependencies import get_graph_ingestion_service

router = APIRouter(prefix="/graph", tags=["Graph Ingestion"])


@router.post(
    "/ingest/triples",
    response_model=TripleIngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_triples_and_blocks(
        request: TripleIngestionRequest,
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
):
    """
    Ingest triples into Neo4j and blocks into PostgreSQL.

    - Stores raw blocks in PostgreSQL for retrieval
    - Creates graph structure in Neo4j from triples
    - Links entities to source blocks
    """
    try:
        response = await service.ingest_triples_and_blocks(request)
        return response
    except Exception as e:
        logger.error(f"Error ingesting triples and blocks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest data: {str(e)}",
        )


@router.get(
    "/blocks/{block_id}",
    status_code=status.HTTP_200_OK,
)
async def get_block(
        block_id: str,
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
) -> Dict[str, Any]:
    """
    Retrieve a specific block by its ID.
    """
    try:
        block = await service.get_block_by_id(block_id)
        if not block:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Block {block_id} not found",
            )
        return block
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving block: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve block: {str(e)}",
        )


@router.get(
    "/documents/{document_id}/blocks",
    status_code=status.HTTP_200_OK,
)
async def get_document_blocks(
        document_id: str,
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
) -> List[Dict[str, Any]]:
    """
    Retrieve all blocks for a specific document.
    """
    try:
        blocks = await service.get_blocks_by_document(document_id)
        return blocks
    except Exception as e:
        logger.error(f"Error retrieving blocks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve blocks: {str(e)}",
        )


@router.get(
    "/documents/{document_id}/graph",
    status_code=status.HTTP_200_OK,
)
async def get_document_graph(
        document_id: str,
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
) -> Dict[str, Any]:
    """
    Retrieve the complete graph structure for a document.
    """
    try:
        graph = await service.get_document_graph(document_id)
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )
        return graph
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document graph: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve graph: {str(e)}",
        )


@router.get(
    "/query/entity/{entity_name}",
    status_code=status.HTTP_200_OK,
)
async def query_by_entity(
        entity_name: str,
        depth: int = Query(2, ge=1, le=5, description="Traversal depth"),
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
) -> Dict[str, Any]:
    """
    Query graph starting from a specific entity.

    Returns the entity and all related entities up to specified depth.
    """
    try:
        results = await service.query_graph_by_entity(entity_name, depth)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity '{entity_name}' not found",
            )
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying entity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query entity: {str(e)}",
        )


@router.get(
    "/query/predicate/{predicate}",
    status_code=status.HTTP_200_OK,
)
async def search_by_predicate(
        predicate: str,
        limit: int = Query(50, ge=1, le=500),
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
) -> List[Dict[str, Any]]:
    """
    Search for all triples with a specific predicate/relationship type.

    Examples: "shall", "may", "refers_to"
    """
    try:
        results = await service.search_by_predicate(predicate, limit)
        return results
    except Exception as e:
        logger.error(f"Error searching by predicate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search: {str(e)}",
        )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
        document_id: str,
        service: GraphIngestionService = Depends(get_graph_ingestion_service),
):
    """
    Delete a document and all associated data.

    Removes:
    - Document node from Neo4j
    - All entity nodes (if not referenced by other documents)
    - All blocks from PostgreSQL
    """
    try:
        await service.delete_document(document_id)
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )