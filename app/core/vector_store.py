import os
from typing import Any, Dict, List, Optional, TypedDict

import lancedb
import numpy as np

from app.core.constants import EMBEDDINGS_TABLE, LANCEDB_PATH
from app.core.hybrid_search import hybrid_search as _hybrid_search


class SearchResult(TypedDict):
    text: str
    score: float


def store_embeddings(file_id: str, embeddings: list[dict]) -> None:
    """Store document embeddings in LanceDB, using inferred schema."""
    import lancedb

    os.makedirs(LANCEDB_PATH, exist_ok=True)
    db = lancedb.connect(LANCEDB_PATH)

    # Prepare data
    data = [
        {
            "text": item["text"],
            "vector": [float(x) for x in item["embedding"]],
            "file_id": file_id,
        }
        for item in embeddings
    ]

    if not data:
        raise ValueError("No embeddings provided")

    if EMBEDDINGS_TABLE not in db.table_names():
        # ✅ Let LanceDB infer schema
        db.create_table(EMBEDDINGS_TABLE, data=data, mode="create")
    else:
        table = db.open_table(EMBEDDINGS_TABLE)
        table.add(data)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def query_embeddings(
    question_embedding: list[float], file_id: Optional[str] = None, top_k: int = 5
) -> list[SearchResult]:
    """Query document embeddings to find most relevant text chunks."""

    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(EMBEDDINGS_TABLE)
    query_vector = np.array(question_embedding, dtype=np.float32)

    if file_id:
        df = table.to_pandas()
        df = df[df["file_id"] == file_id]
        if df.empty:
            return []

        df["relevance"] = df["vector"].apply(
            lambda v: cosine_similarity(query_vector, np.array(v))
        )
        df = df.sort_values(by="relevance", ascending=False).head(top_k)
    else:
        # Use LanceDB native search
        search_query = table.search(query_vector, vector_column_name="vector")
        results = search_query.limit(top_k).to_arrow().to_pandas()

        if results.empty:
            return []

        if "score" in results.columns:
            results["relevance"] = results["score"]
        elif "_distance" in results.columns:
            results["relevance"] = 1 - results["_distance"]
        else:
            raise ValueError(
                "No similarity score found in results. Expected 'score' or '_distance'."
            )
        df = results

    return [
        {"text": str(row["text"]), "score": float(row["relevance"])}
        for _, row in df.iterrows()
    ]


async def hybrid_search(
    query: str,
    question_embedding: List[float],
    file_id: Optional[str] = None,
    k: int = 10,
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """Perform hybrid search combining semantic and keyword search.

    This method delegates to the hybrid_search module which combines:
    - Semantic search via LanceDB vector similarity
    - Keyword search via BM25 (Okapi BM25 algorithm)

    Args:
        query: The search query string (used for BM25 keyword search).
        question_embedding: The embedding vector for the query (used for semantic search).
        file_id: Optional file ID to filter documents.
        k: Number of top results to return. Defaults to 10.
        alpha: Weight for semantic scores (0.5 = equal weight, higher = more semantic).
            Defaults to 0.5. Must be between 0 and 1.

    Returns:
        List of result dictionaries with keys: 'text', 'score'.
        Results are sorted by combined score (descending).

    Note:
        Falls back to vector-only search if BM25 index is unavailable.
    """
    return _hybrid_search(
        query=query,
        question_embedding=question_embedding,
        file_id=file_id,
        k=k,
        alpha=alpha,
    )
