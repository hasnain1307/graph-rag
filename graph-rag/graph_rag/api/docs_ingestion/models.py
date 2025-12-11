from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Triple(BaseModel):
    subject: str = Field(..., description="Subject of the triple")
    predicate: str = Field(..., description="Predicate/relationship type")
    object: str = Field(..., alias="object", description="Object of the triple")
    source: str = Field(..., description="Source block ID")

class Block(BaseModel):
    id: str = Field(..., description="Unique block identifier")
    raw_text: str = Field(..., description="Raw text content of the block")
    terms: List[str] = Field(default_factory=list, description="Extracted terms from block")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TripleIngestionRequest(BaseModel):
    triples: List[Triple] = Field(..., description="List of triples to ingest")
    blocks: List[Block] = Field(..., description="List of blocks to store")
    document_id: Optional[str] = Field(None, description="Optional document ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TripleIngestionResponse(BaseModel):
    document_id: str
    blocks_stored: int
    triples_processed: int
    nodes_created: int
    relationships_created: int
    processing_time: float

class GraphQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int
    query_time: float
