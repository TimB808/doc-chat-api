"""Hybrid search module combining semantic (vector) and keyword (BM25) search.

This module provides hybrid search functionality that combines:
- Semantic search via LanceDB vector similarity
- Keyword search via BM25 (Okapi BM25 algorithm)

Scores are combined using a weighted average:
    final_score = α * semantic_score + (1 - α) * bm25_score

Where α (alpha) defaults to 0.5 and controls the balance between semantic and keyword search.
"""

import logging
import math
import os
import zlib
from typing import TYPE_CHECKING, Any, Optional

import lancedb
import numpy as np
from rank_bm25 import BM25Okapi

from app.core.constants import EMBEDDINGS_TABLE, LANCEDB_PATH

if TYPE_CHECKING:
    from app.core.vector_store import query_embeddings  # noqa: F401

logger = logging.getLogger(__name__)

# Environment variable defaults
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.6"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))

# In-memory cache for BM25 indexes per file_id
# Cache stores (bm25, tokenized_docs, metadata, fingerprint) tuples
_bm25_indexes: dict[
    str, tuple[BM25Okapi, list[list[str]], list[dict[str, Any]], int]
] = {}


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing.

    Simple whitespace tokenization with lowercase normalization.
    For production, consider using more sophisticated tokenizers (spacy, nltk, etc.).
    """
    return text.lower().split()


def _build_bm25_index(
    file_id: Optional[str] = None,
) -> Optional[tuple[BM25Okapi, list[list[str]], list[dict[str, Any]], int]]:
    """Build BM25 index from LanceDB, returning (bm25, tokenized_docs, metadata, fingerprint)."""
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

        texts = df["text"].astype(str).tolist()
        metadata = df.to_dict("records")

        # Fingerprint current content
        fingerprint = zlib.crc32("".join(texts).encode("utf-8")) & 0xFFFFFFFF

        tokenized_docs = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized_docs)

        return bm25, tokenized_docs, metadata, fingerprint

    except Exception as e:
        logger.error(f"Error building BM25 index: {e}", exc_info=True)
        return None


def _get_or_build_bm25_index(
    file_id: Optional[str] = None,
) -> Optional[tuple[BM25Okapi, list[list[str]], list[dict[str, Any]]]]:
    """Return cached (bm25, tokenized_docs, metadata), rebuilding if content changed."""
    cache_key = file_id or "_all"

    # Load current data
    db = lancedb.connect(LANCEDB_PATH)
    if EMBEDDINGS_TABLE not in db.table_names():
        return None
    table = db.open_table(EMBEDDINGS_TABLE)
    df = table.to_pandas()
    if file_id:
        df = df[df["file_id"] == file_id]
    if df.empty:
        return None

    texts = df["text"].astype(str).tolist()
    current_fp = zlib.crc32("".join(texts).encode("utf-8")) & 0xFFFFFFFF

    cached = _bm25_indexes.get(cache_key)
    if cached is not None:
        bm25, tokenized, meta, cached_fp = cached
        if cached_fp == current_fp:
            return bm25, tokenized, meta

    # (Re)build and update cache with fingerprint
    built = _build_bm25_index(file_id)
    if built is None:
        return None
    bm25_obj, tokenized_obj, meta_obj, fp = built
    _bm25_indexes[cache_key] = (bm25_obj, tokenized_obj, meta_obj, fp)
    return bm25_obj, tokenized_obj, meta_obj


def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] via min–max. If all equal → neutral 0.5."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in scores}
    rng = hi - lo
    return {k: (v - lo) / rng for k, v in scores.items()}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# --- MMR utilities ---------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = float(a @ a)
    bb = float(b @ b)
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return float(a @ b) / math.sqrt(aa * bb)


def _mmr_select(
    vectors: list[np.ndarray], query_vec: np.ndarray, k: int, mmr_lambda: float = 0.6
) -> list[int]:
    """
    Maximal Marginal Relevance selection.
    mmr_lambda in [0,1]: higher = more relevance, lower = more diversity.
    Returns indices of selected items.
    """
    n = len(vectors)
    if n == 0:
        return []
    k = min(k, n)

    relevance = [_cosine(query_vec, v) for v in vectors]
    selected: list[int] = []
    used = set()

    # seed with most relevant
    first = max(range(n), key=lambda i: relevance[i])
    selected.append(first)
    used.add(first)

    while len(selected) < k:
        best_i = None
        best_score = -1e9
        for i in range(n):
            if i in used:
                continue
            div = 0.0
            for j in selected:
                div = max(div, _cosine(vectors[i], vectors[j]))
            mmr = mmr_lambda * relevance[i] - (1.0 - mmr_lambda) * div
            if mmr > best_score:
                best_score, best_i = mmr, i
        if best_i is not None:
            used.add(best_i)
            selected.append(best_i)
        else:
            break  # No more candidates
    return selected


def apply_mmr_to_hits(
    hits: list[dict], query_vector: list[float], k: int, mmr_lambda: float = 0.6
) -> list[dict]:
    """
    Non-invasive wrapper to apply MMR to existing hybrid_search() results.

    Expect each hit to contain:
      - 'vector': the embedding (list[float])
      - (optional) 'score': blended score; kept as-is
    Returns the top-k re-ranked hits.
    """
    if not hits:
        return []

    qv = np.array(query_vector, dtype=np.float32)

    vecs: list[np.ndarray] = []
    for h in hits:
        v = h.get("vector")
        arr = np.array(v, dtype=np.float32) if v is not None else None
        # If vector missing or wrong shape, fall back to a zero vector matching qv
        if arr is None or arr.size != qv.size:
            arr = np.zeros_like(qv)
        vecs.append(arr)

    keep = _mmr_select(vecs, qv, k=k, mmr_lambda=mmr_lambda)
    return [hits[i] for i in keep]


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
    k: int = 8,
    alpha: Optional[float] = None,
    use_mmr: bool = False,
    mmr_lambda: Optional[float] = None,
    candidate_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Perform hybrid search combining semantic and keyword search.

    Args:
        query: The search query string (used for BM25 keyword search).
        question_embedding: The embedding vector for the query (used for semantic search).
        file_id: Optional file ID to filter documents.
        k: Number of top results to return.
        alpha: Weight for semantic scores (defaults to HYBRID_ALPHA env var or 0.6).
            Higher = more semantic, lower = more keyword-based.
        use_mmr: If True, apply Maximal Marginal Relevance for result diversification.
        mmr_lambda: MMR lambda parameter (defaults to MMR_LAMBDA env var or 0.6).
            Higher = more relevance, lower = more diversity.
        candidate_k: Number of candidates to consider before MMR (defaults to max(4*k, 24)).
            Only used when use_mmr=True.

    Returns:
        List of result dictionaries with keys: 'text', 'score', and optionally 'file_id'.
        Results are sorted by combined score (descending).

    Note:
        If BM25 index cannot be built, falls back to vector-only search.
        Environment variables MMR_LAMBDA and HYBRID_ALPHA can be used to set defaults.
    """
    # Use environment variable default if alpha not provided
    if alpha is None:
        alpha = HYBRID_ALPHA

    if not 0 <= alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    # If MMR is enabled, use search_hybrid_mmr
    if use_mmr:
        if mmr_lambda is None:
            mmr_lambda = MMR_LAMBDA
        if candidate_k is None:
            candidate_k = max(6 * k, 40)  # give MMR more headroom
        return search_hybrid_mmr(
            query=query,
            question_embedding=question_embedding,
            file_id=file_id,
            k=k,
            alpha=alpha,
            candidate_k=candidate_k,
            mmr_lambda=mmr_lambda,
        )

    # Runtime import to avoid circular dependency
    # TYPE_CHECKING import above is for type annotations only (not executed at runtime)
    from app.core.vector_store import query_embeddings  # noqa: PLC0415

    # Get semantic search results (always available)
    semantic_results = query_embeddings(
        question_embedding=question_embedding,
        file_id=file_id,
        top_k=k * 4,  # Get more candidates before reranking
    )

    if not semantic_results:
        return []

    # Try to get or build BM25 index
    bm25_data = _get_or_build_bm25_index(file_id)
    if bm25_data is None:
        logger.warning("BM25 index not available, falling back to vector-only search")
        # Return top k semantic results with normalized semantic scores only
        sem_raw_fallback = {r["text"]: float(r["score"]) for r in semantic_results}
        sem = _minmax_norm(sem_raw_fallback)
        merged_fallback: list[dict[str, Any]] = []
        for text, s in sem.items():
            final_score = _clamp01(alpha * s + (1 - alpha) * 0.0)
            merged_fallback.append(
                {
                    "id": f"{file_id or '_all'}:{abs(zlib.adler32(text.encode()))}",
                    "text": text,
                    "bm25": 0.0,
                    "semantic": float(s),
                    "final_score": float(final_score),
                }
            )
        merged_fallback.sort(key=lambda r: r["final_score"], reverse=True)
        return merged_fallback[:k]

    bm25, tokenized_docs, metadata = bm25_data

    # Tokenize query for BM25
    tokenized_query = _tokenize(query)

    # Get BM25 scores for all documents in the index (raw, unnormalized)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Create a mapping from text to metadata index
    text_to_index = {str(meta["text"]): idx for idx, meta in enumerate(metadata)}

    # Build raw score dicts keyed by text
    bm25_raw: dict[str, float] = {}
    for idx, meta in enumerate(metadata):
        txt = str(meta["text"])
        try:
            bm25_raw[txt] = float(bm25_scores[idx])
        except Exception:
            bm25_raw[txt] = 0.0

    sem_raw: dict[str, float] = {r["text"]: float(r["score"]) for r in semantic_results}

    # Normalize both to [0, 1]
    bm25_norm = _minmax_norm(bm25_raw)
    sem_norm = _minmax_norm(sem_raw)

    # Merge on union of texts
    ids_set = set(bm25_norm) | set(sem_norm)
    merged: list[dict[str, Any]] = []
    for key_text in ids_set:
        b = float(bm25_norm.get(key_text, 0.0))
        s = float(sem_norm.get(key_text, 0.0))
        final_score = _clamp01(alpha * s + (1 - alpha) * b)
        doc_idx = text_to_index.get(key_text)
        if doc_idx is not None and 0 <= doc_idx < len(metadata):
            fid = str(metadata[doc_idx].get("file_id", file_id or ""))
            stable_id = f"{fid}:{doc_idx}"
        else:
            stable_id = f"{file_id or '_all'}:{abs(zlib.adler32(key_text.encode()))}"

        merged.append(
            {
                "id": stable_id,
                "text": key_text,
                "bm25": b,
                "semantic": s,
                "final_score": float(final_score),
            }
        )

    # Sort by combined score (descending) and return top k
    merged.sort(key=lambda r: r["final_score"], reverse=True)
    return merged[:k]


def search_hybrid_mmr(
    *,
    query: str,
    question_embedding: list[float],
    file_id: Optional[str] = None,
    k: int = 6,
    alpha: float = 0.6,
    candidate_k: int = 30,
    mmr_lambda: float = 0.6,
) -> list[dict]:
    """
    Thin wrapper: call existing hybrid_search() for a larger pool,
    then apply MMR to return a diversified top-k.
    """
    # 1) get a bigger candidate set using current hybrid_search
    candidates = hybrid_search(
        query=query,
        question_embedding=question_embedding,
        file_id=file_id,
        k=max(candidate_k, k),
        alpha=alpha,
    )
    if not candidates:
        return []

    # 2) diversify down to k
    hits = apply_mmr_to_hits(
        hits=candidates,
        query_vector=question_embedding,
        k=k,
        mmr_lambda=mmr_lambda,
    )
    return hits


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
