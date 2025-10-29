"""Hybrid search module combining semantic (vector) and keyword (BM25) search.

This module provides hybrid search functionality that combines:
- Semantic search via LanceDB vector similarity
- Keyword search via BM25 (Okapi BM25 algorithm)

Scores are combined using a weighted average:
    final_score = α * semantic_score + (1 - α) * bm25_score

Where α (alpha) defaults to 0.5 and controls the balance between semantic and keyword search.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import lancedb
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.constants import EMBEDDINGS_TABLE, LANCEDB_PATH

if TYPE_CHECKING:
    from app.core.vector_store import query_embeddings  # noqa: F401

logger = logging.getLogger(__name__)

# In-memory cache for BM25 indexes per file_id
_bm25_indexes: dict[str, tuple[BM25Okapi, list[list[str]], list[dict[str, Any]]]] = {}


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Simple whitespace tokenization with lowercase normalization.
    For production, consider using more sophisticated tokenizers (spacy, nltk, etc.).
    """
    return text.lower().split()


def _build_bm25_index(
    file_id: Optional[str] = None,
) -> Optional[tuple[BM25Okapi, list[list[str]], list[dict[str, Any]]]]:
    """Build BM25 index from documents in LanceDB.

    Args:
        file_id: Optional file ID to filter documents. If None, indexes all documents.

    Returns:
        Tuple of (BM25Okapi instance, tokenized documents, metadata list) or None if no documents found.
    """
    try:
        db = lancedb.connect(LANCEDB_PATH)
        if EMBEDDINGS_TABLE not in db.table_names():
            logger.warning(f"Table {EMBEDDINGS_TABLE} does not exist in LanceDB")
            return None

        table = db.open_table(EMBEDDINGS_TABLE)
        df = table.to_pandas()

        if file_id:
            df = df[df["file_id"] == file_id]
            if df.empty:
                logger.warning(f"No documents found for file_id: {file_id}")
                return None

        if df.empty:
            logger.warning("No documents found in LanceDB")
            return None

        # Extract texts and metadata
        texts = df["text"].astype(str).tolist()
        metadata = df.to_dict("records")

        # Tokenize all documents
        tokenized_docs = [_tokenize(text) for text in texts]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_docs)

        return bm25, tokenized_docs, metadata

    except Exception as e:
        logger.error(f"Error building BM25 index: {e}", exc_info=True)
        return None


def _get_or_build_bm25_index(
    file_id: Optional[str] = None,
) -> Optional[tuple[BM25Okapi, list[list[str]], list[dict[str, Any]]]]:
    """Get cached BM25 index or build a new one.

    Args:
        file_id: Optional file ID to filter documents.

    Returns:
        Tuple of (BM25Okapi instance, tokenized documents, metadata list) or None.
    """
    cache_key = file_id or "_all"
    if cache_key not in _bm25_indexes:
        index_data = _build_bm25_index(file_id)
        if index_data is not None:
            _bm25_indexes[cache_key] = index_data
        else:
            return None
    return _bm25_indexes.get(cache_key)


def _normalize_bm25_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize BM25 scores to [0, 1] range using min-max normalization.

    BM25 scores are unbounded, so we normalize them to match semantic scores
    which are typically in [0, 1] range.

    Args:
        scores: Array of BM25 scores.

    Returns:
        Normalized scores in [0, 1] range.
    """
    if len(scores) == 0:
        return scores
    if scores.max() == scores.min():
        # All scores are the same, return uniform scores
        return np.ones_like(scores) * 0.5
    # Min-max normalization
    return (scores - scores.min()) / (scores.max() - scores.min())


def _combine_scores(
    semantic_scores: np.ndarray, bm25_scores: np.ndarray, alpha: float
) -> np.ndarray:
    """Combine semantic and BM25 scores using weighted average.

    Args:
        semantic_scores: Array of semantic similarity scores (from vector search).
        bm25_scores: Array of normalized BM25 scores.
        alpha: Weight for semantic scores (0 = pure BM25, 1 = pure semantic).

    Returns:
        Combined scores.
    """
    return alpha * semantic_scores + (1 - alpha) * bm25_scores


def hybrid_search(
    query: str,
    question_embedding: list[float],
    file_id: Optional[str] = None,
    k: int = 10,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """Perform hybrid search combining semantic and keyword search.

    Args:
        query: The search query string (used for BM25 keyword search).
        question_embedding: The embedding vector for the query (used for semantic search).
        file_id: Optional file ID to filter documents.
        k: Number of top results to return.
        alpha: Weight for semantic scores (0.5 = equal weight, higher = more semantic).

    Returns:
        List of result dictionaries with keys: 'text', 'score', and optionally 'file_id'.
        Results are sorted by combined score (descending).

    Note:
        If BM25 index cannot be built, falls back to vector-only search.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    # Runtime import to avoid circular dependency
    # TYPE_CHECKING import above is for type annotations only (not executed at runtime)
    from app.core.vector_store import query_embeddings  # noqa: PLC0415

    # Get semantic search results (always available)
    semantic_results = query_embeddings(
        question_embedding=question_embedding,
        file_id=file_id,
        top_k=k * 2,  # Get more candidates for reranking
    )

    if not semantic_results:
        return []

    # Try to get or build BM25 index
    bm25_data = _get_or_build_bm25_index(file_id)
    if bm25_data is None:
        logger.warning("BM25 index not available, falling back to vector-only search")
        # Return top k semantic results, converting SearchResult to Dict
        return [
            {"text": result["text"], "score": result["score"]}
            for result in semantic_results[:k]
        ]

    bm25, tokenized_docs, metadata = bm25_data

    # Tokenize query for BM25
    tokenized_query = _tokenize(query)

    # Get BM25 scores for all documents in the index
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_scores_normalized = _normalize_bm25_scores(np.array(bm25_scores))

    # Create a mapping from text to metadata index
    text_to_index = {str(meta["text"]): idx for idx, meta in enumerate(metadata)}

    # Combine semantic and BM25 scores
    combined_results = []
    seen_texts = set()

    for sem_result in semantic_results:
        text = sem_result["text"]
        sem_score = sem_result["score"]

        # Get BM25 score for this text
        if text in text_to_index:
            doc_idx = text_to_index[text]
            bm25_score = float(bm25_scores_normalized[doc_idx])
        else:
            # Text not in BM25 index (shouldn't happen, but handle gracefully)
            logger.debug(f"Text not found in BM25 index: {text[:50]}...")
            bm25_score = 0.0

        # Combine scores
        combined_score = _combine_scores(
            np.array([sem_score]), np.array([bm25_score]), alpha
        )[0]

        # Avoid duplicates
        if text not in seen_texts:
            combined_results.append(
                {
                    "text": text,
                    "score": float(combined_score),
                }
            )
            seen_texts.add(text)

    # Sort by combined score (descending) and return top k
    combined_results.sort(key=lambda x: x["score"], reverse=True)  # type: ignore[arg-type,return-value]
    return combined_results[:k]


def clear_bm25_cache(file_id: Optional[str] = None) -> None:
    """Clear cached BM25 index.

    Args:
        file_id: Optional file ID to clear specific index. If None, clears all caches.
    """
    if file_id:
        cache_key = file_id
        if cache_key in _bm25_indexes:
            del _bm25_indexes[cache_key]
    else:
        _bm25_indexes.clear()
    logger.info(f"Cleared BM25 cache for: {file_id or 'all'}")
