import os
from functools import partial
from typing import Any, Optional, TypedDict

import anyio
import lancedb
import numpy as np

from app.core.constants import EMBEDDINGS_TABLE, LANCEDB_PATH
from app.core.hybrid_search import (
    HYBRID_ALPHA,
    MMR_LAMBDA,
)
from app.core.hybrid_search import (
    search_hybrid_mmr as _search_hybrid_mmr,
)


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
) -> list[dict[str, Any]]:
    """Query document embeddings to find most relevant text chunks."""
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(EMBEDDINGS_TABLE)
    query_vector = np.array(question_embedding, dtype=np.float32)

    if file_id:
        df = table.to_pandas()
        df = df[df["file_id"] == file_id]
        if df.empty:
            return []

        # Cosine similarity in [-1,1] -> map to [0,1] for consistency
        df["relevance_raw"] = df["vector"].apply(
            lambda v: cosine_similarity(query_vector, np.array(v))
        )
        df["relevance"] = (df["relevance_raw"] + 1.0) / 2.0
        df = df.sort_values(by="relevance", ascending=False).head(top_k)

    else:
        # Use LanceDB native search
        results = (
            table.search(query_vector, vector_column_name="vector")
            .limit(top_k)
            .to_arrow()
            .to_pandas()
        )
        if results.empty:
            return []

        if "score" in results.columns:
            # Assume score is a similarity (higher is better)
            results["relevance"] = results["score"].astype(float)

        elif "_distance" in results.columns:
            # Convert distance -> bounded similarity in (0,1]
            results["_distance"] = results["_distance"].astype(float)
            results["relevance"] = 1.0 / (1.0 + results["_distance"])

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
    question_embedding: list[float],
    file_id: Optional[str] = None,
    k: int = 10,
    alpha: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Perform hybrid + MMR search (threaded for non-blocking behaviour).

    Args:
        query: The search query string.
        question_embedding: The embedding vector for the query.
        file_id: Optional file ID to filter documents.
        k: Number of top results to return.
        alpha: Weight for semantic scores (defaults to HYBRID_ALPHA env var or 0.6).

    Environment variables:
        MMR_LAMBDA: MMR lambda parameter (defaults to 0.6).
        HYBRID_ALPHA: Default alpha value (defaults to 0.6).
    """
    # Use environment variable default if alpha not provided
    if alpha is None:
        alpha = HYBRID_ALPHA

    fn = partial(
        _search_hybrid_mmr,
        query=query,
        question_embedding=question_embedding,
        file_id=file_id,
        k=k,
        alpha=alpha,
        candidate_k=max(4 * k, 24),
        mmr_lambda=MMR_LAMBDA,
    )
    return await anyio.to_thread.run_sync(fn)
