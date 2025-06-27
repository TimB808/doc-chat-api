import os
from typing import List, Dict, Optional, TypedDict

import lancedb
import numpy as np
import pandas as pd

# Constants
LANCEDB_PATH = "data/lancedb"
EMBEDDINGS_TABLE = "document_embeddings"

class SearchResult(TypedDict):
    text: str
    score: float

def store_embeddings(file_id: str, embeddings: List[Dict]) -> None:
    """Store document embeddings in LanceDB, using inferred schema."""
    import os
    import lancedb

    os.makedirs(LANCEDB_PATH, exist_ok=True)
    db = lancedb.connect(LANCEDB_PATH)

    # Prepare data
    data = [
        {
            "text": item["text"],
            "vector": [float(x) for x in item["embedding"]],
            "file_id": file_id
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



def query_embeddings(question_embedding: List[float], file_id: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
    """Query document embeddings to find most relevant text chunks."""
    import lancedb
    import numpy as np

    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(EMBEDDINGS_TABLE)

    query_vector = [float(x) for x in question_embedding]  # Force raw list to avoid shape issues
    search_query = table.search(query_vector, vector_column_name="vector")

    if file_id:
        search_query = search_query.where(f"file_id = '{file_id}'")

    results = (
        search_query
        .limit(top_k)
        .to_arrow()
        .to_pandas()
    )

    if results.empty:
        return []

    # Use whichever column LanceDB provides for similarity
    if "score" in results.columns:
        results["relevance"] = results["score"]
    elif "_distance" in results.columns:
        results["relevance"] = 1 - results["_distance"]
    else:
        raise ValueError("No similarity score found in results. Expected 'score' or '_distance'.")

    return [
        {"text": str(row["text"]), "score": float(row["relevance"])}
        for _, row in results.iterrows()
    ]
