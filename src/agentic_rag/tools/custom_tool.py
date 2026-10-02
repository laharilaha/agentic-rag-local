import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient, models


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""

    query: str = Field(
        ...,
        description="Query to search the document."
    )


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        super().__init__()

        self.file_path = file_path
        self.collection_name = "demo_collection"

        # In-memory Qdrant database
        self.client = QdrantClient(":memory:")

        # Same embedding model used by Chonkie
        self.embedding_model = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        ).embedding_model

        self._process_document()

    def _extract_text(self) -> str:
        """Extract text from PDF."""

        md = MarkItDown()
        result = md.convert(self.file_path)

        return result.text_content

    def _create_chunks(self, raw_text: str):
        """Create semantic chunks."""

        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )

        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process document and store embeddings in Qdrant."""

        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)

        documents = [chunk.text for chunk in chunks]

        metadata = [
            {
                "source": os.path.basename(self.file_path)
            }
            for _ in chunks
        ]

        ids = list(range(len(documents)))

        # Create embeddings
        embeddings = self.embedding_model.embed_batch(documents)

        # Create Qdrant collection
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        # Insert vectors
        points = [
            models.PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={
                    "document": documents[i],
                    "metadata": metadata[i],
                },
            )
            for i in range(len(documents))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _run(self, query: str) -> str:
        """Search the document."""

        query_vector = self.embedding_model.embed(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=5,
        )

        docs = [
            point.payload["document"]
            for point in results.points
        ]

        return "\n___\n".join(docs)