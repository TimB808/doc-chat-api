import os
from typing import Optional, TypedDict

import lancedb
import numpy as np

# Constants
LANCEDB_PATH = "data/lancedb"
EMBEDDINGS_TABLE = "document_embeddings"


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
