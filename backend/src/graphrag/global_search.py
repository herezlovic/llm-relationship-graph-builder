"""
Full-scale GraphRAG global search (Local-to-Global map-reduce).

Paper community levels (C0 root → C3 leaf) are mapped onto Neo4j Leiden levels
where level 0 is the leaf (entities attach via IN_COMMUNITY) and higher levels
are parents via PARENT_COMMUNITY.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graphrag.helpers import (
    chunk_summaries,
    estimate_tokens,
    format_summaries_for_prompt,
    parse_map_response,
    resolve_paper_community_level,
)
from src.graphrag.prompts import (
    MAP_HUMAN_PROMPT,
    MAP_SYSTEM_PROMPT,
    REDUCE_HUMAN_PROMPT,
    REDUCE_SYSTEM_PROMPT,
)
from src.llm import get_llm
from src.shared.common_fn import get_value_from_env

logger = logging.getLogger(__name__)

# Re-export for callers/tests
__all_helpers__ = [
    "resolve_paper_community_level",
    "chunk_summaries",
    "parse_map_response",
]

CHAT_GLOBAL_MAP_REDUCE_MODE = "global_map_reduce"
CHAT_GLOBAL_C0_MODE = "global_c0"
CHAT_GLOBAL_C1_MODE = "global_c1"
CHAT_GLOBAL_C2_MODE = "global_c2"
CHAT_GLOBAL_C3_MODE = "global_c3"
# Paper comparison condition: map-reduce over source texts (not community summaries).
CHAT_TS_MAP_REDUCE_MODE = "ts_map_reduce"

GRAPH_RAG_GLOBAL_MODES = {
    CHAT_GLOBAL_MAP_REDUCE_MODE,
    CHAT_GLOBAL_C0_MODE,
    CHAT_GLOBAL_C1_MODE,
    CHAT_GLOBAL_C2_MODE,
    CHAT_GLOBAL_C3_MODE,
    CHAT_TS_MAP_REDUCE_MODE,
}

MODE_TO_PAPER_LEVEL = {
    CHAT_GLOBAL_C0_MODE: 0,
    CHAT_GLOBAL_C1_MODE: 1,
    CHAT_GLOBAL_C2_MODE: 2,
    CHAT_GLOBAL_C3_MODE: 3,
    CHAT_GLOBAL_MAP_REDUCE_MODE: None,  # auto / env default
    CHAT_TS_MAP_REDUCE_MODE: None,
}

DEFAULT_MAP_CHUNK_TOKENS = 3000
DEFAULT_REDUCE_TOKEN_BUDGET = 8000
DEFAULT_MAP_WORKERS = 8
DEFAULT_TS_CHUNK_LIMIT = 500

GET_MAX_COMMUNITY_LEVEL = """
MATCH (c:__Community__)
WHERE c.summary IS NOT NULL
RETURN coalesce(max(c.level), 0) AS max_level
"""

GET_COMMUNITY_SUMMARIES_AT_LEVEL = """
MATCH (c:__Community__)
WHERE c.summary IS NOT NULL AND c.level = $level
RETURN elementId(c) AS id,
       c.id AS community_id,
       c.title AS title,
       c.summary AS summary,
       c.level AS level,
       coalesce(c.community_rank, 0) AS community_rank,
       coalesce(c.weight, 0) AS weight
ORDER BY community_rank DESC, weight DESC
"""

GET_ALL_COMMUNITY_SUMMARIES = """
MATCH (c:__Community__)
WHERE c.summary IS NOT NULL
RETURN elementId(c) AS id,
       c.id AS community_id,
       c.title AS title,
       c.summary AS summary,
       c.level AS level,
       coalesce(c.community_rank, 0) AS community_rank,
       coalesce(c.weight, 0) AS weight
ORDER BY c.level DESC, community_rank DESC, weight DESC
"""

GET_SOURCE_CHUNK_TEXTS = """
MATCH (c:Chunk)
WHERE c.text IS NOT NULL AND trim(c.text) <> ''
OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
RETURN elementId(c) AS id,
       coalesce(c.id, elementId(c)) AS community_id,
       coalesce(d.fileName, 'chunk') AS title,
       c.text AS summary,
       0 AS level,
       coalesce(c.position, 0) AS community_rank,
       size(c.text) AS weight
ORDER BY coalesce(d.fileName, ''), coalesce(c.position, 0)
LIMIT $limit
"""


def _fetch_community_summaries(graph, neo4j_level: Optional[int]) -> List[Dict[str, Any]]:
    if neo4j_level is None:
        rows = graph.query(GET_ALL_COMMUNITY_SUMMARIES)
    else:
        rows = graph.query(GET_COMMUNITY_SUMMARIES_AT_LEVEL, params={"level": int(neo4j_level)})
    return [dict(row) for row in rows if row.get("summary")]


def _fetch_source_chunk_texts(graph) -> List[Dict[str, Any]]:
    limit = get_value_from_env("GRAPHRAG_TS_CHUNK_LIMIT", DEFAULT_TS_CHUNK_LIMIT, "int")
    rows = graph.query(GET_SOURCE_CHUNK_TEXTS, params={"limit": int(limit)})
    return [dict(row) for row in rows if row.get("summary")]


def _get_max_community_level(graph) -> int:
    rows = graph.query(GET_MAX_COMMUNITY_LEVEL)
    if not rows:
        return 0
    return int(rows[0].get("max_level") or 0)


def _build_map_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MAP_SYSTEM_PROMPT),
            ("human", MAP_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def _build_reduce_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REDUCE_SYSTEM_PROMPT),
            ("human", REDUCE_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def _map_chunk(chain, question: str, chunk: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        raw = chain.invoke(
            {
                "question": question,
                "community_summaries": format_summaries_for_prompt(chunk),
            }
        )
        helpfulness, answer = parse_map_response(raw)
        if helpfulness <= 0 or not answer:
            return None
        return {
            "helpfulness": helpfulness,
            "answer": answer,
            "community_ids": [c.get("id") for c in chunk if c.get("id")],
            "summaries_used": len(chunk),
        }
    except Exception as exc:
        logger.error("GraphRAG map step failed: %s", exc)
        return None


def run_global_map_reduce(
    graph,
    model: str,
    question: str,
    paper_level: Optional[int] = None,
    source: str = "communities",
) -> Dict[str, Any]:
    """
    Execute Local-to-Global map-reduce over community summaries or source texts.

    ``source``:
      - ``communities`` (default): GraphRAG C0–C3 / global map-reduce
      - ``chunks`` / ``ts``: paper TS condition — map-reduce over Chunk texts
    """
    start = time.time()
    llm, model_name, _ = get_llm(model)
    source_key = (source or "communities").strip().lower()
    use_ts = source_key in {"chunks", "ts", "text", "source_texts"}

    max_level = 0
    neo_level = None
    if use_ts:
        summaries = _fetch_source_chunk_texts(graph)
        empty_msg = "No source chunk texts are available. Extract documents first."
    else:
        max_level = _get_max_community_level(graph)
        neo_level = resolve_paper_community_level(paper_level, max_level)
        summaries = _fetch_community_summaries(graph, neo_level)
        empty_msg = (
            "No community summaries are available. Run the enable_communities post-processing job first."
        )

    if not summaries:
        return {
            "answer": empty_msg,
            "model": model_name,
            "communities": [],
            "partial_answers": [],
            "paper_level": paper_level,
            "neo4j_level": neo_level,
            "max_neo4j_level": max_level,
            "source": "ts" if use_ts else "communities",
            "response_time": time.time() - start,
            "total_tokens": 0,
        }

    map_chunk_tokens = get_value_from_env("GRAPHRAG_MAP_CHUNK_TOKENS", DEFAULT_MAP_CHUNK_TOKENS, "int")
    reduce_budget = get_value_from_env("GRAPHRAG_REDUCE_TOKEN_BUDGET", DEFAULT_REDUCE_TOKEN_BUDGET, "int")
    max_workers = get_value_from_env("GRAPHRAG_MAP_WORKERS", DEFAULT_MAP_WORKERS, "int")

    chunks = chunk_summaries(summaries, map_chunk_tokens)
    map_chain = _build_map_chain(llm)
    partials: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(_map_chunk, map_chain, question, chunk) for chunk in chunks]
        for future in as_completed(futures):
            result = future.result()
            if result:
                partials.append(result)

    if not partials:
        return {
            "answer": (
                "I could not find helpful source-text evidence to answer that question."
                if use_ts
                else "I could not find helpful community evidence to answer that question."
            ),
            "model": model_name,
            "communities": [s.get("id") for s in summaries if s.get("id")],
            "partial_answers": [],
            "paper_level": paper_level,
            "neo4j_level": neo_level,
            "max_neo4j_level": max_level,
            "source": "ts" if use_ts else "communities",
            "response_time": time.time() - start,
            "total_tokens": 0,
        }

    partials.sort(key=lambda item: item["helpfulness"], reverse=True)

    selected: List[Dict[str, Any]] = []
    used_tokens = 0
    for partial in partials:
        tokens = estimate_tokens(partial["answer"])
        if selected and used_tokens + tokens > reduce_budget:
            break
        selected.append(partial)
        used_tokens += tokens

    reduce_payload = "\n\n".join(
        f"[helpfulness={p['helpfulness']}]\n{p['answer']}" for p in selected
    )
    reduce_chain = _build_reduce_chain(llm)
    final_answer = reduce_chain.invoke({"question": question, "partial_answers": reduce_payload})

    community_ids = []
    seen = set()
    for partial in selected:
        for cid in partial.get("community_ids") or []:
            if cid not in seen:
                seen.add(cid)
                community_ids.append(cid)

    return {
        "answer": final_answer.strip(),
        "model": model_name,
        "communities": community_ids,
        "partial_answers": selected,
        "paper_level": paper_level,
        "neo4j_level": neo_level,
        "max_neo4j_level": max_level,
        "map_chunks": len(chunks),
        "source": "ts" if use_ts else "communities",
        "response_time": time.time() - start,
        "total_tokens": 0,
    }


def _paper_level_for_mode(mode: str) -> Optional[int]:
    if mode in MODE_TO_PAPER_LEVEL:
        configured = MODE_TO_PAPER_LEVEL[mode]
        if configured is not None:
            return configured
        env_level = get_value_from_env("GRAPHRAG_DEFAULT_PAPER_LEVEL", None)
        if env_level is None or str(env_level).strip() == "":
            return None
        try:
            return int(env_level)
        except (TypeError, ValueError):
            return None
    return None


def process_graphrag_global_response(
    model: str,
    graph,
    question: str,
    messages: List[Any],
    history,
    mode: str,
):
    """Chat adapter for GraphRAG map-reduce modes (including paper TS)."""
    try:
        paper_level = _paper_level_for_mode(mode)
        source = "ts" if mode == CHAT_TS_MAP_REDUCE_MODE else "communities"
        result = run_global_map_reduce(
            graph, model, question, paper_level=paper_level, source=source
        )
        content = result["answer"]
        ai_response = AIMessage(content=content)
        messages.append(ai_response)

        # Reuse existing history summarization pattern without importing circular deps heavily.
        try:
            from src.QA_integration import summarize_and_log

            llm, _, _ = get_llm(model)
            summarization_thread = threading.Thread(target=summarize_and_log, args=(history, messages, llm))
            summarization_thread.start()
        except Exception as hist_exc:
            logger.warning("Could not start chat history summarization: %s", hist_exc)

        community_details = [{"id": cid} for cid in result.get("communities") or []]
        return {
            "session_id": "",
            "message": content,
            "info": {
                "sources": [],
                "model": result.get("model"),
                "nodedetails": {"chunkdetails": [], "entitydetails": [], "communitydetails": community_details},
                "total_tokens": result.get("total_tokens", 0),
                "response_time": round(result.get("response_time", 0), 2),
                "mode": mode,
                "entities": {"entityids": [], "relationshipids": []},
                "metric_details": {
                    "question": question,
                    "contexts": [p.get("answer") for p in result.get("partial_answers") or []],
                    "answer": content,
                },
                "graphrag": {
                    "paper_level": result.get("paper_level"),
                    "neo4j_level": result.get("neo4j_level"),
                    "max_neo4j_level": result.get("max_neo4j_level"),
                    "map_chunks": result.get("map_chunks"),
                    "partial_answers": result.get("partial_answers"),
                    "source": result.get("source"),
                },
            },
            "user": "chatbot",
        }
    except Exception as exc:
        logger.exception("GraphRAG global response failed at %s: %s", datetime.now(), exc)
        return {
            "session_id": "",
            "message": "Something went wrong while running GraphRAG global search.",
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
