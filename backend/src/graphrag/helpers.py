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


def _format_claim_line(claim: Dict[str, Any]) -> str:
    subject = claim.get("subject") or ""
    object_ = claim.get("object") or ""
    claim_type = claim.get("type") or "claim"
    description = claim.get("description") or ""
    source_span = claim.get("source_span") or ""
    start_date = claim.get("start_date") or ""
    end_date = claim.get("end_date") or ""
    entity_id = claim.get("entity_id") or ""
    entity_ids = claim.get("entity_ids") or ([] if not entity_id else [entity_id])
    linked = f", entities: {', '.join(str(e) for e in entity_ids)}" if entity_ids else ""
    span = f', source_span: "{source_span}"' if source_span else ""
    dates = ""
    if start_date or end_date:
        dates = f", dates: {start_date or '?'}..{end_date or '?'}"
    return (
        f"claim: ({subject})-[{claim_type}]->({object_}), "
        f"description: {description}{span}{dates}{linked}\n"
    )


def prepare_community_string(community_data: Dict[str, Any], max_chars: int = 12000) -> str:
    """
    Degree-prioritized community prompt text (shared by communities + tests).

    GraphRAG leaf prioritization: relationships by combined degree, with claim
    covariates for involved entities included when present.
    """
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

    claims = list(community_data.get("claims") or [])
    claims_by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for claim in claims:
        entity_ids = list(claim.get("entity_ids") or [])
        if claim.get("entity_id") and claim.get("entity_id") not in entity_ids:
            entity_ids.append(claim["entity_id"])
        if not entity_ids:
            # Fallback: index under subject so unlinked claims still surface
            subject = (claim.get("subject") or "").strip()
            if subject:
                entity_ids = [subject]
        for eid in entity_ids:
            claims_by_entity.setdefault(str(eid), []).append(claim)

    relationships = list(community_data.get("rels") or [])
    relationships.sort(key=lambda rel: rel.get("combined_degree") or 0, reverse=True)

    relationships_description = "Relationships are:\n"
    claims_description = "Claims (covariates) are:\n"
    included_claim_keys = set()
    used_chars = len(nodes_description) + len(relationships_description) + len(claims_description)

    def _claim_key(claim: Dict[str, Any]) -> str:
        return str(claim.get("id") or _format_claim_line(claim))

    for rel in relationships:
        start_node = rel["start"]
        end_node = rel["end"]
        relationship_type = rel["type"]
        relationship_description = (
            f", description: {rel['description']}" if rel.get("description") else ""
        )
        candidate = f"({start_node})-[:{relationship_type}]->({end_node}){relationship_description}\n"
        if used_chars + len(candidate) > max_chars:
            break
        relationships_description += candidate
        used_chars += len(candidate)

        # After each edge, include linked covariates for source/target (paper order).
        for eid in (start_node, end_node):
            for claim in claims_by_entity.get(str(eid), []):
                key = _claim_key(claim)
                if key in included_claim_keys:
                    continue
                claim_line = _format_claim_line(claim)
                if used_chars + len(claim_line) > max_chars:
                    continue
                claims_description += claim_line
                used_chars += len(claim_line)
                included_claim_keys.add(key)

    # Include any remaining claims not attached via the relationship walk.
    for claim in claims:
        key = _claim_key(claim)
        if key in included_claim_keys:
            continue
        claim_line = _format_claim_line(claim)
        if used_chars + len(claim_line) > max_chars:
            break
        claims_description += claim_line
        used_chars += len(claim_line)
        included_claim_keys.add(key)

    parts = [nodes_description, relationships_description]
    if included_claim_keys or claims:
        # Keep section header even when empty so prompts stay stable when claims exist but were truncated.
        if included_claim_keys:
            parts.append(claims_description)
        elif claims:
            parts.append(claims_description + "(none within token budget)\n")
    return "\n".join(parts)
