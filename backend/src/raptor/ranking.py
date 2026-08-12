"""Pure RAPTOR ranking / selection helpers (no LLM or Neo4j imports)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CHARS_PER_TOKEN_ESTIMATE = 4


def has_embedding(emb: Any) -> bool:
    """True when emb is a non-empty vector (list/tuple/ndarray)."""
    if emb is None:
        return False
    try:
        length = len(emb)
    except TypeError:
        return False
    return length > 0


def cosine_similarity(query_vec: Sequence[float], doc_vec: Sequence[float]) -> float:
    q = np.asarray(query_vec, dtype=np.float64)
    d = np.asarray(doc_vec, dtype=np.float64)
    if q.size == 0 or d.size == 0 or q.shape != d.shape:
        return 0.0
    denom = np.linalg.norm(q) * np.linalg.norm(d)
    if denom == 0:
        return 0.0
    return float(np.dot(q, d) / denom)


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN_ESTIMATE)


def rank_nodes(query_embedding: Sequence[float], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for node in nodes:
        emb = node.get("embedding")
        if not has_embedding(emb):
            continue
        score = cosine_similarity(query_embedding, emb)
        item = dict(node)
        item["score"] = score
        ranked.append(item)
    ranked.sort(key=lambda n: n["score"], reverse=True)
    return ranked


def select_collapsed_nodes(
    ranked: List[Dict[str, Any]],
    top_k: int,
    max_context_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Greedy selection over cosine-ranked RAPTOR nodes.

    ``top_k`` is a hard maximum (aligned with tree-traversal layer width).
    Token budget is soft: the first node is always kept; later nodes may slightly
    overshoot the budget only while still filling toward ``top_k``.
    """
    top_k = max(1, int(top_k))
    max_context_tokens = max(1, int(max_context_tokens))
    selected: List[Dict[str, Any]] = []
    used = 0
    for node in ranked:
        if len(selected) >= top_k:
            break
        tokens = estimate_tokens(node.get("text") or "")
        if selected and used + tokens > max_context_tokens:
            # Prefer packing a smaller later node over stopping early when under top_k,
            # but never keep adding once the budget is already exhausted.
            if used >= max_context_tokens:
                break
            continue
        selected.append(node)
        used += tokens
    return selected
