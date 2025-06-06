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
    """Store document embeddings in LanceDB.
    
    Args:
        file_id: Unique identifier for the document
        embeddings: List of dicts containing text chunks and their embeddings
                   Each dict has format: {'text': str, 'embedding': List[float]}
    """
    # Create data directory if it doesn't exist
    os.makedirs(LANCEDB_PATH, exist_ok=True)
    
    # Connect to LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    
    # Convert embeddings to format expected by LanceDB
    data = [
        {
            "text": item["text"],
            "embedding": np.array(item["embedding"], dtype=np.float32),
            "file_id": file_id
        }
        for item in embeddings
    ]
    
    try:
        # Try to get existing table
        table = db.open_table(EMBEDDINGS_TABLE)
        # Add new embeddings to existing table
        table.add(data)
    except FileNotFoundError:
        # Table doesn't exist, create it
        db.create_table(
            EMBEDDINGS_TABLE,
            data=data,
            mode="create"
        )

def query_embeddings(question_embedding: List[float], file_id: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
    """Query document embeddings to find most relevant text chunks.
    
    Args:
        question_embedding: Embedding vector of the question
        file_id: Optional file ID to filter results by specific document
        top_k: Number of results to return (default: 5)
        
    Returns:
        List of dicts containing text chunks and their similarity scores
    """
    # Connect to LanceDB
    db = lancedb.connect(LANCEDB_PATH)
    
    # Open embeddings table
    table = db.open_table(EMBEDDINGS_TABLE)
    
    # Convert embedding to numpy array
    query_vector = np.array(question_embedding, dtype=np.float32)
    
    # Build search query
    search_query = table.search(query_vector)
    
    # Add file_id filter if provided
    if file_id:
        search_query = search_query.where(f"file_id = '{file_id}'")
    
    # Execute search
    results = (
        search_query
        .limit(top_k)
        .select(["text", "_distance"])
        .to_pandas()
    )
    
    # Handle empty results
    if results.empty:
        return []
    
    # Convert distance scores to similarity scores (1 = most similar, 0 = least similar)
    results["score"] = 1 - results["_distance"]
    
    # Convert to list of dicts and ensure Python native types for JSON serialization
    return [
        {
            "text": str(row["text"]),
            "score": float(row["score"])
        }
        for _, row in results.iterrows()
    ]
