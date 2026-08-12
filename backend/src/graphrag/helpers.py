"""Pure helpers for GraphRAG map-reduce (no LLM/Neo4j imports)."""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

CHARS_PER_TOKEN_ESTIMATE = 4


def resolve_paper_community_level(paper_level: Optional[int], max_neo4j_level: int) -> Optional[int]:
    """
    Map paper C0–C3 onto Neo4j Leiden levels.

    Paper C0 = root (fewest) → Neo4j max level
    Paper C3 = leaf (most) → Neo4j level 0
    """
    if paper_level is None:
        return None
    if max_neo4j_level < 0:
        return 0
    paper_level = max(0, min(3, int(paper_level)))
    neo_level = max_neo4j_level - paper_level
    if neo_level < 0:
        neo_level = 0
    if neo_level > max_neo4j_level:
        neo_level = max_neo4j_level
    return neo_level


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def chunk_summaries(summaries: List[Dict[str, Any]], max_tokens: int) -> List[List[Dict[str, Any]]]:
    shuffled = list(summaries)
    random.shuffle(shuffled)
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_tokens = 0
    for item in shuffled:
        text = item.get("summary") or ""
        tokens = estimate_tokens(text)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += tokens
    if current:
        chunks.append(current)
    return chunks


def parse_map_response(raw: str) -> Tuple[int, str]:
    helpfulness = 0
    answer = raw.strip()
    help_match = re.search(r"helpfulness\s*:\s*(\d{1,3})", raw, flags=re.IGNORECASE)
    if help_match:
        helpfulness = max(0, min(100, int(help_match.group(1))))
    answer_match = re.search(r"answer\s*:\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    return helpfulness, answer


def format_summaries_for_prompt(summaries: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, item in enumerate(summaries, start=1):
        title = item.get("title") or "Untitled"
        summary = item.get("summary") or ""
        community_id = item.get("community_id") or item.get("id") or i
        blocks.append(f"[{i}] community={community_id} title={title}\n{summary}")
    return "\n\n----\n\n".join(blocks)


def prepare_community_string(community_data: Dict[str, Any], max_chars: int = 12000) -> str:
    """Degree-prioritized community prompt text (shared by communities + tests)."""
    nodes = list(community_data.get("nodes") or [])
    nodes.sort(key=lambda node: node.get("degree") or 0, reverse=True)
    nodes_description = "Nodes are:\n"
    for node in nodes:
        node_id = node["id"]
        node_type = node.get("type")
        node_description = f", description: {node['description']}" if node.get("description") else ""
        degree = node.get("degree")
        degree_text = f", degree: {degree}" if degree is not None else ""
        nodes_description += f"id: {node_id}, type: {node_type}{node_description}{degree_text}\n"

    relationships = list(community_data.get("rels") or [])
    relationships.sort(key=lambda rel: rel.get("combined_degree") or 0, reverse=True)

    relationships_description = "Relationships are:\n"
    for rel in relationships:
        start_node = rel["start"]
        end_node = rel["end"]
        relationship_type = rel["type"]
        relationship_description = (
            f", description: {rel['description']}" if rel.get("description") else ""
        )
        candidate = f"({start_node})-[:{relationship_type}]->({end_node}){relationship_description}\n"
        if len(nodes_description) + len(relationships_description) + len(candidate) > max_chars:
            break
        relationships_description += candidate
    return nodes_description + "\n" + relationships_description
