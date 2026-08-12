"""RAPTOR query strategies: tree traversal and collapsed tree."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.raptor.prompts import RAPTOR_QA_HUMAN, RAPTOR_QA_SYSTEM
from src.shared.common_fn import get_value_from_env, load_embedding_model

logger = logging.getLogger(__name__)

CHAT_RAPTOR_COLLAPSED_MODE = "raptor_collapsed"
CHAT_RAPTOR_TREE_MODE = "raptor_tree"

RAPTOR_MODES = {
    CHAT_RAPTOR_COLLAPSED_MODE,
    CHAT_RAPTOR_TREE_MODE,
}

FETCH_ALL_RAPTOR_NODES = """
MATCH (n:__RaptorNode__)
WHERE n.embedding IS NOT NULL AND n.text IS NOT NULL
RETURN n.id AS id,
       n.text AS text,
       n.layer AS layer,
       n.is_leaf AS is_leaf,
       n.embedding AS embedding,
       n.source_chunk_ids AS source_chunk_ids,
       n.file_names AS file_names
"""

FETCH_ROOT_RAPTOR_NODES = """
MATCH (n:__RaptorNode__)
WHERE n.embedding IS NOT NULL AND NOT EXISTS { (n)<-[:HAS_CHILD]-() }
RETURN n.id AS id,
       n.text AS text,
       n.layer AS layer,
       n.is_leaf AS is_leaf,
       n.embedding AS embedding,
       n.source_chunk_ids AS source_chunk_ids,
       n.file_names AS file_names
"""

FETCH_CHILDREN = """
MATCH (parent:__RaptorNode__ {id: $parent_id})-[:HAS_CHILD]->(child:__RaptorNode__)
WHERE child.embedding IS NOT NULL
RETURN child.id AS id,
       child.text AS text,
       child.layer AS layer,
       child.is_leaf AS is_leaf,
       child.embedding AS embedding,
       child.source_chunk_ids AS source_chunk_ids,
       child.file_names AS file_names
"""

CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_TOP_K = 5
DEFAULT_CONTEXT_TOKENS = 2000


def _cosine_similarity(query_vec: Sequence[float], doc_vec: Sequence[float]) -> float:
    q = np.asarray(query_vec, dtype=np.float64)
    d = np.asarray(doc_vec, dtype=np.float64)
    denom = np.linalg.norm(q) * np.linalg.norm(d)
    if denom == 0:
        return 0.0
    return float(np.dot(q, d) / denom)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN_ESTIMATE)


def _rank_nodes(query_embedding: Sequence[float], nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for node in nodes:
        emb = node.get("embedding")
        if not emb:
            continue
        score = _cosine_similarity(query_embedding, emb)
        item = dict(node)
        item["score"] = score
        ranked.append(item)
    ranked.sort(key=lambda n: n["score"], reverse=True)
    return ranked


def collapsed_tree_retrieve(
    graph,
    query_embedding: Sequence[float],
    top_k: Optional[int] = None,
    max_context_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    top_k = top_k or get_value_from_env("RAPTOR_TOP_K", DEFAULT_TOP_K, "int")
    max_context_tokens = max_context_tokens or get_value_from_env(
        "RAPTOR_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS, "int"
    )
    nodes = [dict(row) for row in graph.query(FETCH_ALL_RAPTOR_NODES)]
    ranked = _rank_nodes(query_embedding, nodes)

    selected: List[Dict[str, Any]] = []
    used = 0
    for node in ranked:
        tokens = _estimate_tokens(node.get("text") or "")
        if selected and used + tokens > max_context_tokens:
            if len(selected) >= top_k:
                break
            # Still allow filling top_k with smaller nodes if budget remains tight
            if used >= max_context_tokens:
                break
        selected.append(node)
        used += tokens
        if len(selected) >= max(top_k, 1) and used >= max_context_tokens:
            break
    return selected


def tree_traversal_retrieve(
    graph,
    query_embedding: Sequence[float],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    top_k = top_k or get_value_from_env("RAPTOR_TOP_K", DEFAULT_TOP_K, "int")
    roots = [dict(row) for row in graph.query(FETCH_ROOT_RAPTOR_NODES)]
    if not roots:
        # Fall back to collapsed retrieval if hierarchy links are missing.
        return collapsed_tree_retrieve(graph, query_embedding, top_k=top_k)

    selected: List[Dict[str, Any]] = []
    current_layer_nodes = _rank_nodes(query_embedding, roots)[:top_k]
    selected.extend(current_layer_nodes)

    visited_layers = 0
    max_layers = get_value_from_env("RAPTOR_MAX_LAYERS", 5, "int")
    while current_layer_nodes and visited_layers < max_layers:
        children: List[Dict[str, Any]] = []
        for parent in current_layer_nodes:
            child_rows = graph.query(FETCH_CHILDREN, params={"parent_id": parent["id"]})
            children.extend(dict(row) for row in child_rows)
        if not children:
            break
        current_layer_nodes = _rank_nodes(query_embedding, children)[:top_k]
        selected.extend(current_layer_nodes)
        visited_layers += 1

    # De-duplicate by id, keep highest score
    best: Dict[str, Dict[str, Any]] = {}
    for node in selected:
        node_id = node["id"]
        if node_id not in best or node.get("score", 0) > best[node_id].get("score", 0):
            best[node_id] = node
    ordered = sorted(best.values(), key=lambda n: n.get("score", 0), reverse=True)
    return ordered


def _format_context(nodes: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, node in enumerate(nodes, start=1):
        layer = node.get("layer")
        score = round(float(node.get("score") or 0), 4)
        files = ", ".join(node.get("file_names") or []) or "unknown"
        blocks.append(
            f"[{i}] layer={layer} score={score} files={files}\n{node.get('text') or ''}"
        )
    return "\n\n----\n\n".join(blocks)


def run_raptor_query(
    graph,
    model: str,
    question: str,
    mode: str,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.time()
    embeddings_model, _ = load_embedding_model(embedding_provider, embedding_model)
    query_embedding = embeddings_model.embed_query(question)

    if mode == CHAT_RAPTOR_TREE_MODE:
        nodes = tree_traversal_retrieve(graph, query_embedding)
        strategy = "tree_traversal"
    else:
        nodes = collapsed_tree_retrieve(graph, query_embedding)
        strategy = "collapsed_tree"

    if not nodes:
        return {
            "answer": "No RAPTOR index found. Run the enable_raptor post-processing job first.",
            "model": model,
            "nodes": [],
            "strategy": strategy,
            "response_time": time.time() - start,
            "total_tokens": 0,
        }

    llm, model_name, _ = get_llm(model)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAPTOR_QA_SYSTEM),
            ("human", RAPTOR_QA_HUMAN),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": _format_context(nodes)}).strip()

    return {
        "answer": answer,
        "model": model_name,
        "nodes": nodes,
        "strategy": strategy,
        "response_time": time.time() - start,
        "total_tokens": 0,
    }


def process_raptor_response(
    model: str,
    graph,
    question: str,
    messages: List[Any],
    history,
    mode: str,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    try:
        result = run_raptor_query(
            graph,
            model,
            question,
            mode,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        content = result["answer"]
        messages.append(AIMessage(content=content))

        try:
            from src.QA_integration import summarize_and_log

            llm, _, _ = get_llm(model)
            summarization_thread = threading.Thread(target=summarize_and_log, args=(history, messages, llm))
            summarization_thread.start()
        except Exception as hist_exc:
            logger.warning("Could not start chat history summarization: %s", hist_exc)

        sources = sorted(
            {
                file_name
                for node in result.get("nodes") or []
                for file_name in (node.get("file_names") or [])
                if file_name
            }
        )
        community_like = [
            {
                "id": node.get("id"),
                "score": node.get("score"),
                "layer": node.get("layer"),
            }
            for node in result.get("nodes") or []
        ]

        return {
            "session_id": "",
            "message": content,
            "info": {
                "sources": sources,
                "model": result.get("model"),
                "nodedetails": {
                    "chunkdetails": [],
                    "entitydetails": [],
                    "communitydetails": community_like,
                },
                "total_tokens": result.get("total_tokens", 0),
                "response_time": round(result.get("response_time", 0), 2),
                "mode": mode,
                "entities": {"entityids": [], "relationshipids": []},
                "metric_details": {
                    "question": question,
                    "contexts": [n.get("text") for n in result.get("nodes") or []],
                    "answer": content,
                },
                "raptor": {
                    "strategy": result.get("strategy"),
                    "nodes_used": len(result.get("nodes") or []),
                },
            },
            "user": "chatbot",
        }
    except Exception as exc:
        logger.exception("RAPTOR response failed at %s: %s", datetime.now(), exc)
        return {
            "session_id": "",
            "message": "Something went wrong while running RAPTOR retrieval.",
            "info": {
                "sources": [],
                "model": "",
                "nodedetails": [],
                "total_tokens": 0,
                "response_time": 0,
                "mode": mode,
                "entities": [],
                "metric_details": {},
                "error": f"{type(exc).__name__}: {str(exc)}",
            },
            "user": "chatbot",
        }
