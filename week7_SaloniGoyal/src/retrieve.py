from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

RAG_INDEX_FILENAME = "rag_index.pkl"

def tokenize(text: str) -> list[str]:
    """
    Convert text into searchable lowercase tokens.
    """
    return re.findall(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]+",
        text.lower(),
    )

def normalize_vector(
    vector: dict[str, float],
) -> dict[str, float]:
    """
    Normalize a vector using L2 normalization.
    """
    magnitude = math.sqrt(
        sum(value * value for value in vector.values())
    )

    if magnitude == 0:
        return {}

    return {
        term: value / magnitude
        for term, value in vector.items()
    }

def dot_product(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    """
    Compute cosine similarity dot product.
    """
    # Iterate over the smaller vector for efficiency
    if len(left) > len(right):
        left, right = right, left

    return sum(
        value * right.get(term, 0.0)
        for term, value in left.items()
    )

def normalize_chunk(
    chunk: str | dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """
    Ensure all chunks follow the same structure.
    """
    if isinstance(chunk, dict):
        return chunk

    return {
        "id": f"chunk-{index}",
        "text": chunk,
        "source": "memory",
    }

class TfidfVectorStore:
    """
    Lightweight local TF-IDF vector search engine.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        idf_scores: dict[str, float],
        vectors: list[dict[str, float]],
        metadata: dict[str, Any] | None = None,
    ) -> None:

        self.chunks = chunks
        self.idf_scores = idf_scores
        self.vectors = vectors
        self.metadata = metadata or {}

    @property
    def size(self) -> int:
        """
        Total indexed chunks.
        """
        return len(self.chunks)

    @classmethod
    def build(
        cls,
        chunks: Iterable[str | dict[str, Any]],
    ) -> "TfidfVectorStore":
        """
        Build a TF-IDF vector store from text chunks.
        """

        normalized_chunks = [
            normalize_chunk(chunk, idx)
            for idx, chunk in enumerate(chunks)
        ]

        # Count tokens in each chunk
        token_counts = [
            Counter(
                tokenize(chunk.get("text", ""))
            )
            for chunk in normalized_chunks
        ]

        
        document_frequency: Counter[str] = Counter()

        for counts in token_counts:
            document_frequency.update(counts.keys())

        total_documents = max(
            len(normalized_chunks),
            1,
        )

        
        idf_scores = {
            term: math.log(
                (1 + total_documents)
                / (1 + frequency)
            ) + 1
            for term, frequency in document_frequency.items()
        }

        vectors = [
            cls.vectorize_counts(
                counts,
                idf_scores,
            )
            for counts in token_counts
        ]

        return cls(
            normalized_chunks,
            idf_scores,
            vectors,
        )

    @staticmethod
    def vectorize_counts(
        counts: Counter[str],
        idf_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Convert token counts into normalized TF-IDF vectors.
        """
        weighted_terms = {
            term: (1 + math.log(count))
            * idf_scores[term]
            for term, count in counts.items()
            if term in idf_scores and count > 0
        }

        return normalize_vector(weighted_terms)

    def vectorize_query(
        self,
        query: str,
    ) -> dict[str, float]:
        """
        Create a TF-IDF vector for a search query.
        """
        query_counts = Counter(
            tokenize(query)
        )

        return self.vectorize_counts(
            query_counts,
            self.idf_scores,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks.
        """
        if top_k <= 0:
            return []

        query_vector = self.vectorize_query(query)

        if not query_vector:
            return []

        scored_chunks: list[
            tuple[float, dict[str, Any]]
        ] = []

        for chunk, vector in zip(
            self.chunks,
            self.vectors,
        ):
            similarity = dot_product(
                query_vector,
                vector,
            )

            if similarity > 0:
                scored_chunks.append(
                    (similarity, chunk)
                )

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        results: list[dict[str, Any]] = []

        for score, chunk in scored_chunks[:top_k]:
            results.append(
                {
                    **chunk,
                    "score": score,
                }
            )

        return results

    def save(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save vector store data to disk.
        """
        output_path = Path(output_dir)

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        if metadata is not None:
            self.metadata = metadata

        index_path = (
            output_path / RAG_INDEX_FILENAME
        )

        with index_path.open("wb") as file:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "idf_scores": self.idf_scores,
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                },
                file,
            )

        
        with (
            output_path / "chunks.pkl"
        ).open("wb") as file:
            pickle.dump(
                self.chunks,
                file,
            )

        return index_path

    @classmethod
    def load(
        cls,
        input_path: str | Path,
    ) -> "TfidfVectorStore":
        """
        Load a saved vector store.
        """
        path = Path(input_path)

        index_path = (
            path / RAG_INDEX_FILENAME
            if path.is_dir()
            else path
        )

        with index_path.open("rb") as file:
            payload = pickle.load(file)

        return cls(
            chunks=payload["chunks"],
            idf_scores=payload["idf_scores"],
            vectors=payload["vectors"],
            metadata=payload.get(
                "metadata",
                {},
            ),
        )


def build_vector_store(
    chunks: Iterable[str | dict[str, Any]],
    output_dir: str | Path = "vector_store",
    metadata: dict[str, Any] | None = None,
) -> TfidfVectorStore:
    """
    Build and save a vector store.
    """
    store = TfidfVectorStore.build(chunks)

    store.save(
        output_dir,
        metadata=metadata,
    )

    return store


def load_vector_store(
    input_path: str | Path = "vector_store",
) -> TfidfVectorStore:
    """
    Load an existing vector store.
    """
    return TfidfVectorStore.load(input_path)


def retrieve_relevant_chunks(
    query: str,
    chunks_or_store:
    list[str]
    | list[dict[str, Any]]
    | TfidfVectorStore,
    top_k: int = 3,
) -> list[str] | list[dict[str, Any]]:
    """
    Retrieve the most relevant chunks
    for a given query.
    """

    
    if isinstance(
        chunks_or_store,
        TfidfVectorStore,
    ):
        return chunks_or_store.search(
            query,
            top_k=top_k,
        )

    input_was_plain_text = all(
        isinstance(chunk, str)
        for chunk in chunks_or_store
    )

    
    temporary_store = TfidfVectorStore.build(
        chunks_or_store
    )

    results = temporary_store.search(
        query,
        top_k=top_k,
    )


    if input_was_plain_text:
        return [
            result["text"]
            for result in results
        ]

    return results