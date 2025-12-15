from typing import List, Dict, Any, Optional
from neo4j import Driver
from datetime import datetime
import uuid
import json
from loguru import logger
from injector import inject, singleton
import asyncpg

from graph_rag.api.docs_ingestion.models import (
    Triple,
    Block,
    TripleIngestionRequest,
    TripleIngestionResponse,
)


@singleton
class GraphIngestionService:
    @inject
    def __init__(self, neo4j_driver: Driver, pgvector_pool: asyncpg.Pool):
        self.neo4j_driver = neo4j_driver
        self.pgvector_pool = pgvector_pool

    async def ingest_triples_and_blocks(
            self, request: TripleIngestionRequest
    ) -> TripleIngestionResponse:
        """
        Ingest triples into Neo4j and blocks into PostgreSQL.

        Args:
            request: Triple ingestion request containing triples and blocks

        Returns:
            TripleIngestionResponse with ingestion statistics
        """
        start_time = datetime.now()
        document_id = request.document_id or str(uuid.uuid4())

        # Store blocks in PostgreSQL
        blocks_stored = await self._store_blocks(request.blocks, document_id)

        # Ingest triples into Neo4j
        nodes_created, relationships_created = await self._ingest_triples(
            request.triples, document_id, request.metadata
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return TripleIngestionResponse(
            document_id=document_id,
            blocks_stored=blocks_stored,
            triples_processed=len(request.triples),
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            processing_time=processing_time,
        )

    async def _store_blocks(self, blocks: List[Block], document_id: str) -> int:
        """
        Store raw blocks in PostgreSQL.
        """
        async with self.pgvector_pool.acquire() as conn:
            # Create table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_blocks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    terms TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create index on document_id
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_blocks_document_id 
                ON document_blocks(document_id)
            """)

            # Insert blocks
            blocks_stored = 0
            for block in blocks:
                try:
                    await conn.execute(
                        """
                        INSERT INTO document_blocks 
                        (id, document_id, raw_text, terms, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE SET
                            raw_text = EXCLUDED.raw_text,
                            terms = EXCLUDED.terms,
                            metadata = EXCLUDED.metadata,
                            updated_at = NOW()
                        """,
                        block.id,
                        document_id,
                        block.raw_text,
                        block.terms,
                        json.dumps(block.metadata or {}),
                    )
                    blocks_stored += 1
                except Exception as e:
                    logger.error(f"Error storing block {block.id}: {e}")

            return blocks_stored

    async def _ingest_triples(
            self, triples: List[Triple], document_id: str, metadata: Dict[str, Any]
    ) -> tuple[int, int]:
        """
        Ingest triples into Neo4j graph.
        Returns (nodes_created, relationships_created)
        """
        nodes_created = 0
        relationships_created = 0

        with self.neo4j_driver.session() as session:
            # Create document node
            session.run(
                """
                MERGE (d:Document {id: $id})
                ON CREATE SET 
                    d.created_at = datetime(),
                    d.metadata = $metadata
                ON MATCH SET
                    d.updated_at = datetime()
                """,
                id=document_id,
                metadata=json.dumps(metadata or {}),
            )
            nodes_created += 1

            # Process each triple
            for triple in triples:
                try:
                    # Create subject node
                    session.run(
                        """
                        MERGE (s:Entity {name: $subject})
                        ON CREATE SET 
                            s.id = randomUUID(),
                            s.created_at = datetime(),
                            s.type = 'Generic'
                        """,
                        subject=triple.subject,
                    )

                    # Create object node
                    session.run(
                        """
                        MERGE (o:Entity {name: $object})
                        ON CREATE SET 
                            o.id = randomUUID(),
                            o.created_at = datetime(),
                            o.type = 'Generic'
                        """,
                        object=triple.object,
                    )

                    nodes_created += 2  # Subject and object

                    # Create relationship between subject and object
                    # Clean predicate to make valid Neo4j relationship type
                    rel_type = self._sanitize_relationship_type(triple.predicate)

                    result = session.run(
                        f"""
                        MATCH (s:Entity {{name: $subject}})
                        MATCH (o:Entity {{name: $object}})
                        MERGE (s)-[r:{rel_type}]->(o)
                        ON CREATE SET 
                            r.created_at = datetime(),
                            r.predicate = $predicate,
                            r.source = $source,
                            r.document_id = $document_id
                        RETURN r
                        """,
                        subject=triple.subject,
                        object=triple.object,
                        predicate=triple.predicate,
                        source=triple.source,
                        document_id=document_id,
                    )

                    if result.single():
                        relationships_created += 1

                    # Link entities to document
                    session.run(
                        """
                        MATCH (d:Document {id: $document_id})
                        MATCH (s:Entity {name: $subject})
                        MERGE (d)-[r:CONTAINS]->(s)
                        ON CREATE SET r.created_at = datetime()
                        """,
                        document_id=document_id,
                        subject=triple.subject,
                    )

                    session.run(
                        """
                        MATCH (d:Document {id: $document_id})
                        MATCH (o:Entity {name: $object})
                        MERGE (d)-[r:CONTAINS]->(o)
                        ON CREATE SET r.created_at = datetime()
                        """,
                        document_id=document_id,
                        object=triple.object,
                    )

                except Exception as e:
                    logger.error(f"Error processing triple: {e}")
                    logger.error(f"Triple: {triple}")

        return nodes_created, relationships_created

    def _sanitize_relationship_type(self, predicate: str) -> str:
        """
        Sanitize predicate to make valid Neo4j relationship type.
        Neo4j relationship types must be uppercase and contain only letters, numbers, and underscores.
        """
        # Replace spaces and special chars with underscore
        sanitized = predicate.upper().replace(" ", "_")
        # Remove any characters that aren't alphanumeric or underscore
        sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
        # Remove consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")

        return sanitized or "RELATED_TO"

    async def get_blocks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all blocks for a document from PostgreSQL.
        """
        async with self.pgvector_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, raw_text, terms, metadata, created_at, updated_at
                FROM document_blocks
                WHERE document_id = $1
                ORDER BY id
                """,
                document_id,
            )

            return [
                {
                    "id": row["id"],
                    "raw_text": row["raw_text"],
                    "terms": row["terms"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }
                for row in rows
            ]

    async def get_block_by_id(self, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific block by ID.
        """
        async with self.pgvector_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, document_id, raw_text, terms, metadata, created_at, updated_at
                FROM document_blocks
                WHERE id = $1
                """,
                block_id,
            )

            if not row:
                return None

            return {
                "id": row["id"],
                "document_id": row["document_id"],
                "raw_text": row["raw_text"],
                "terms": row["terms"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }

    async def query_graph_by_entity(
            self, entity_name: str, depth: int = 2
    ) -> Dict[str, Any]:
        """
        Query graph starting from an entity and traverse relationships.
        """
        with self.neo4j_driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (e:Entity {{name: $entity_name}})-[r*1..{depth}]-(related)
                WITH e, relationships(path) as rels, related
                RETURN e,
                       collect(DISTINCT related) as related_entities,
                       [rel in rels | {{
                           type: type(rel),
                           predicate: rel.predicate,
                           source: rel.source
                       }}] as relationships
                LIMIT 100
                """,
                entity_name=entity_name,
            )

            record = result.single()
            if not record:
                return {}

            return {
                "entity": dict(record["e"]),
                "related_entities": [dict(e) for e in record["related_entities"]],
                "relationships": record["relationships"],
            }

    async def get_document_graph(self, document_id: str) -> Dict[str, Any]:
        """
        Get complete graph for a document.
        """
        with self.neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $document_id})-[:CONTAINS]->(e:Entity)
                OPTIONAL MATCH (e)-[r]-(related:Entity)
                WHERE (d)-[:CONTAINS]->(related)
                RETURN d,
                       collect(DISTINCT e) as entities,
                       collect(DISTINCT r) as relationships,
                       collect(DISTINCT related) as related_entities
                """,
                document_id=document_id,
            )

            record = result.single()
            if not record:
                return {}

            return {
                "document": dict(record["d"]),
                "entities": [dict(e) for e in record["entities"]],
                "relationships": [
                    {
                        "type": type(r).__name__,
                        "properties": dict(r),
                    }
                    for r in record["relationships"]
                    if r is not None
                ],
            }

    async def search_by_predicate(
            self, predicate: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for all triples with a specific predicate.
        """
        rel_type = self._sanitize_relationship_type(predicate)

        with self.neo4j_driver.session() as session:
            result = session.run(
                f"""
                MATCH (s:Entity)-[r:{rel_type}]->(o:Entity)
                RETURN s.name as subject, 
                       r.predicate as predicate,
                       o.name as object,
                       r.source as source
                LIMIT $limit
                """,
                limit=limit,
            )

            return [
                {
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"],
                    "source": record["source"],
                }
                for record in result
            ]

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document from both Neo4j and PostgreSQL.
        """
        # Delete from Neo4j
        with self.neo4j_driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:CONTAINS]->(e:Entity)
                WHERE NOT EXISTS {
                    MATCH (other:Document)-[:CONTAINS]->(e)
                    WHERE other.id <> $document_id
                }
                DETACH DELETE d, e
                """,
                document_id=document_id,
            )

        # Delete from PostgreSQL
        async with self.pgvector_pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM document_blocks WHERE document_id = $1",
                document_id,
            )

        return True