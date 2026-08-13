"""
GraphRAG element-summary consolidation (homogeneous clustering).

Paper step: instance-level element descriptions → embed → cosine/KNN clusters of
homogeneous summaries → LLM summarize each cluster into a single element_summary
block used later by community summarization.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

ELEMENT_SUMMARY_DEFAULT_MODEL = "openai_gpt_5_mini"
DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_MAX_CLUSTER_SIZE = 12
DEFAULT_WORKERS = 6

FETCH_ENTITY_DESCRIPTIONS = """
MATCH (e:__Entity__)
WHERE e.description IS NOT NULL AND trim(e.description) <> ''
RETURN elementId(e) AS element_id,
       e.id AS id,
       e.description AS description,
       coalesce(e.element_summary, null) AS element_summary,
       [l IN labels(e) WHERE l <> '__Entity__'][0] AS type
"""

WRITE_ELEMENT_SUMMARIES = """
UNWIND $rows AS row
MATCH (e) WHERE elementId(e) = row.element_id
SET e.element_summary = row.element_summary,
    e.element_summary_cluster_size = row.cluster_size
"""

ELEMENT_SUMMARY_SYSTEM = (
    "You consolidate near-duplicate entity descriptions into one clear element summary. "
    "Preserve distinct facts; remove redundancy. No preamble."
)

ELEMENT_SUMMARY_HUMAN = """Entity type: {entity_type}
Entity ids in this homogeneous cluster:
{entity_ids}

Instance descriptions:
{descriptions}

Write a single consolidated element summary (natural language, 2-6 sentences).
"""


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


def cluster_homogeneous_indices(
    embeddings: List[Sequence[float]],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
) -> List[List[int]]:
    """
    Connected-component clustering over cosine similarity >= threshold.

    Large components are greedily partitioned into chunks of ``max_cluster_size``
    (sorted by index) so LLM context stays bounded.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    threshold = float(threshold)
    max_cluster_size = max(1, int(max_cluster_size))
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: List[List[int]] = []
    for members in groups.values():
        members.sort()
        if len(members) <= max_cluster_size:
            clusters.append(members)
            continue
        for start in range(0, len(members), max_cluster_size):
            clusters.append(members[start : start + max_cluster_size])
    clusters.sort(key=lambda c: c[0])
    return clusters


def select_description_for_prompt(entity: Dict[str, Any]) -> str:
    """Prefer consolidated element_summary when present."""
    summary = (entity.get("element_summary") or "").strip()
    if summary:
        return summary
    return (entity.get("description") or "").strip()


def _build_summary_chain(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ELEMENT_SUMMARY_SYSTEM),
            ("human", ELEMENT_SUMMARY_HUMAN),
        ]
    )
    return prompt | llm | StrOutputParser()


def _summarize_cluster(chain, entities: List[Dict[str, Any]]) -> Optional[str]:
    if not entities:
        return None
    if len(entities) == 1:
        return select_description_for_prompt(entities[0]) or None
    descriptions = []
    for i, ent in enumerate(entities, start=1):
        descriptions.append(f"[{i}] id={ent.get('id')}: {ent.get('description') or ''}")
    try:
        raw = chain.invoke(
            {
                "entity_type": entities[0].get("type") or "Entity",
                "entity_ids": ", ".join(str(e.get("id")) for e in entities),
                "descriptions": "\n".join(descriptions),
            }
        )
        text = (raw or "").strip()
        return text or None
    except Exception as exc:
        logger.error("element summary LLM failed: %s", exc)
        # Fallback: concatenate truncated descriptions
        parts = [select_description_for_prompt(e) for e in entities]
        return " ".join(p for p in parts if p)[:4000] or None


def consolidate_element_summaries(
    graph,
    model: str = ELEMENT_SUMMARY_DEFAULT_MODEL,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Embed entity descriptions, cluster homogeneous ones, and write element_summary.

    ``email`` is accepted for token-tracking parity with communities/raptor (reserved).
    """
    del email  # reserved for future token tracking parity
    from src.llm import get_llm
    from src.shared.common_fn import get_value_from_env, load_embedding_model

    threshold = get_value_from_env(
        "GRAPHRAG_ELEMENT_SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD, "float"
    )
    max_cluster_size = get_value_from_env(
        "GRAPHRAG_ELEMENT_MAX_CLUSTER_SIZE", DEFAULT_MAX_CLUSTER_SIZE, "int"
    )
    max_workers = get_value_from_env("GRAPHRAG_ELEMENT_SUMMARY_WORKERS", DEFAULT_WORKERS, "int")

    rows = [dict(r) for r in graph.query(FETCH_ENTITY_DESCRIPTIONS)]
    if not rows:
        logger.info("consolidate_element_summaries: no entity descriptions found")
        return {"entities": 0, "clusters": 0, "updated": 0}

    embeddings_model, _ = load_embedding_model(embedding_provider, embedding_model)
    vectors: List[List[float]] = []
    valid_rows: List[Dict[str, Any]] = []
    for row in rows:
        text = (row.get("description") or "").strip()
        if not text:
            continue
        try:
            vectors.append(list(embeddings_model.embed_query(text)))
            valid_rows.append(row)
        except Exception as exc:
            logger.warning("Failed to embed entity %s: %s", row.get("id"), exc)

    if not valid_rows:
        return {"entities": 0, "clusters": 0, "updated": 0}

    clusters = cluster_homogeneous_indices(
        vectors, threshold=float(threshold), max_cluster_size=int(max_cluster_size)
    )
    llm, _, _ = get_llm(model)
    chain = _build_summary_chain(llm)

    updates: List[Dict[str, Any]] = []

    def _work(cluster_indices: List[int]) -> List[Dict[str, Any]]:
        members = [valid_rows[i] for i in cluster_indices]
        summary = _summarize_cluster(chain, members)
        if not summary:
            return []
        return [
            {
                "element_id": m["element_id"],
                "element_summary": summary,
                "cluster_size": len(members),
            }
            for m in members
        ]

    # Only LLM-summarize multi-member clusters in parallel; singles copy description.
    multi = [c for c in clusters if len(c) > 1]
    singles = [c for c in clusters if len(c) == 1]

    for cluster in singles:
        member = valid_rows[cluster[0]]
        text = (member.get("description") or "").strip()
        if text:
            updates.append(
                {
                    "element_id": member["element_id"],
                    "element_summary": text,
                    "cluster_size": 1,
                }
            )

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(_work, cluster) for cluster in multi]
        for future in as_completed(futures):
            try:
                updates.extend(future.result() or [])
            except Exception as exc:
                logger.error("element summary cluster worker failed: %s", exc)

    if updates:
        graph.query(WRITE_ELEMENT_SUMMARIES, params={"rows": updates})

    result = {
        "entities": len(valid_rows),
        "clusters": len(clusters),
        "multi_member_clusters": len(multi),
        "updated": len(updates),
    }
    logger.info("consolidate_element_summaries: %s", result)
    return result
