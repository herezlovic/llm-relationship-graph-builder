"""Claim covariate extraction and Neo4j persistence for GraphRAG.

Extracts claims (subject, object, type, description, source span, optional dates)
linked to entities found in chunk text. Designed as an optional post-processing
step: LLM failures return empty results and do not break the extract pipeline.

Pure helpers (parse/persist) avoid LLM imports so unit tests need no live deps.
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

CLAIM_NODE_LABEL = "__Claim__"
CLAIM_EXTRACTION_DEFAULT_MODEL = "openai_gpt_5_mini"
MAX_CLAIM_WORKERS = 8

CREATE_CLAIM_CONSTRAINT = (
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (c:{CLAIM_NODE_LABEL}) REQUIRE c.id IS UNIQUE"
)

FETCH_CHUNKS_WITH_ENTITIES = """
MATCH (c:Chunk)-[:HAS_ENTITY]->(e:__Entity__)
WHERE c.text IS NOT NULL AND trim(c.text) <> ''
WITH c, collect(DISTINCT {id: e.id, description: coalesce(e.description, '')}) AS entities
RETURN elementId(c) AS element_id,
       c.id AS chunk_id,
       c.text AS text,
       entities
ORDER BY c.position
"""

STORE_CLAIMS = """
UNWIND $rows AS row
MERGE (claim:__Claim__ {id: row.id})
SET claim.subject = row.subject,
    claim.object = row.object,
    claim.type = row.type,
    claim.description = row.description,
    claim.source_span = row.source_span,
    claim.start_date = row.start_date,
    claim.end_date = row.end_date,
    claim.status = row.status
WITH claim, row
OPTIONAL MATCH (chunk:Chunk) WHERE elementId(chunk) = row.chunk_element_id
FOREACH (_ IN CASE WHEN chunk IS NULL THEN [] ELSE [1] END |
  MERGE (chunk)-[:HAS_CLAIM]->(claim)
)
WITH claim, row
OPTIONAL MATCH (e:__Entity__)
WHERE size(coalesce(row.entity_ids, [])) > 0 AND e.id IN row.entity_ids
FOREACH (_ IN CASE WHEN e IS NULL THEN [] ELSE [1] END |
  MERGE (e)-[:HAS_CLAIM]->(claim)
)
RETURN count(DISTINCT claim) AS stored
"""

CLAIM_EXTRACTION_SYSTEM = """You extract factual claims (covariates) linked to known entities in a text chunk.
Only extract claims that are explicitly supported by the chunk text.
Respond with zero or more claim blocks in exactly this format (no preamble):

===CLAIM===
subject: <entity or actor name>
object: <entity, value, or target of the claim>
type: <short claim type, e.g. allegation, acquisition, affiliation, event>
description: <one or two sentence claim description>
source_span: <exact short quote from the chunk>
start_date: <YYYY-MM-DD or empty>
end_date: <YYYY-MM-DD or empty>
status: <suspected|known|disputed|empty>

If there are no claims, respond with exactly: NO_CLAIMS
"""

CLAIM_EXTRACTION_HUMAN = """Known entities in this chunk (prefer these as subject/object when relevant):
{entities}

Chunk text:
{chunk_text}
"""

# Lookahead must not use (?====...) — (?= consumes the first '=', leaving ==CLAIM===.
_CLAIM_BLOCK_RE = re.compile(
    r"===CLAIM===\s*(.*?)(?=\n===CLAIM===|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_FIELD_RE = re.compile(
    r"^(subject|object|type|description|source_span|start_date|end_date|status)\s*:\s*(.*)$",
    re.IGNORECASE,
)


def make_claim_id(
    subject: str,
    object_: str,
    claim_type: str,
    description: str,
    source_span: str = "",
) -> str:
    """Deterministic claim id for idempotent MERGE."""
    payload = "|".join(
        [
            (subject or "").strip().lower(),
            (object_ or "").strip().lower(),
            (claim_type or "").strip().lower(),
            (description or "").strip().lower(),
            (source_span or "").strip().lower(),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]
    return f"claim-{digest}"


def parse_claims_response(raw: str) -> List[Dict[str, str]]:
    """Parse LLM claim blocks into structured dicts. Safe on empty/malformed input."""
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    if re.search(r"^\s*NO_CLAIMS\s*$", text, flags=re.IGNORECASE):
        return []

    blocks = _CLAIM_BLOCK_RE.findall(text)
    if not blocks and "subject:" in text.lower():
        blocks = [text]

    claims: List[Dict[str, str]] = []
    for block in blocks:
        fields = {
            "subject": "",
            "object": "",
            "type": "",
            "description": "",
            "source_span": "",
            "start_date": "",
            "end_date": "",
            "status": "",
        }
        for line in block.splitlines():
            match = _FIELD_RE.match(line.strip())
            if not match:
                continue
            key = match.group(1).lower()
            value = match.group(2).strip()
            fields[key] = value
        if not fields["subject"] and not fields["description"]:
            continue
        if not fields["type"]:
            fields["type"] = "claim"
        claims.append(fields)
    return claims


def resolve_entity_ids_for_claim(
    claim: Dict[str, str],
    entities: Sequence[Dict[str, Any]],
) -> List[str]:
    """Match claim subject/object to known entity ids (case-insensitive substring/equality)."""
    entity_ids: List[str] = []
    subject = (claim.get("subject") or "").strip().lower()
    object_ = (claim.get("object") or "").strip().lower()
    for ent in entities or []:
        eid = ent.get("id")
        if not eid:
            continue
        eid_l = str(eid).strip().lower()
        if subject and (eid_l == subject or subject in eid_l or eid_l in subject):
            entity_ids.append(str(eid))
            continue
        if object_ and (eid_l == object_ or object_ in eid_l or eid_l in object_):
            entity_ids.append(str(eid))
    seen = set()
    ordered: List[str] = []
    for eid in entity_ids:
        if eid not in seen:
            seen.add(eid)
            ordered.append(eid)
    return ordered


def claims_to_persist_rows(
    claims: Sequence[Dict[str, str]],
    *,
    chunk_element_id: Optional[str] = None,
    entities: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Normalize parsed claims into Neo4j MERGE rows."""
    rows: List[Dict[str, Any]] = []
    for claim in claims:
        subject = (claim.get("subject") or "").strip()
        object_ = (claim.get("object") or "").strip()
        claim_type = (claim.get("type") or "claim").strip()
        description = (claim.get("description") or "").strip()
        source_span = (claim.get("source_span") or "").strip()
        if not subject and not description:
            continue
        claim_id = claim.get("id") or make_claim_id(
            subject, object_, claim_type, description, source_span
        )
        entity_ids = list(claim.get("entity_ids") or [])
        if not entity_ids and entities is not None:
            entity_ids = resolve_entity_ids_for_claim(claim, entities)
        rows.append(
            {
                "id": claim_id,
                "subject": subject,
                "object": object_,
                "type": claim_type,
                "description": description,
                "source_span": source_span,
                "start_date": (claim.get("start_date") or "").strip() or None,
                "end_date": (claim.get("end_date") or "").strip() or None,
                "status": (claim.get("status") or "").strip() or None,
                "chunk_element_id": chunk_element_id,
                "entity_ids": entity_ids,
            }
        )
    return rows


def persist_claims(graph, rows: Sequence[Dict[str, Any]]) -> int:
    """Idempotently MERGE __Claim__ nodes and HAS_CLAIM links. Returns stored count."""
    if not rows:
        return 0
    graph.query(CREATE_CLAIM_CONSTRAINT)
    result = graph.query(STORE_CLAIMS, params={"rows": list(rows)})
    if not result:
        return len(rows)
    stored = result[0].get("stored")
    return int(stored) if stored is not None else len(rows)


def _format_entities_for_prompt(entities: Sequence[Dict[str, Any]]) -> str:
    if not entities:
        return "(none)"
    lines = []
    for ent in entities:
        eid = ent.get("id") or ""
        desc = ent.get("description") or ""
        if desc:
            lines.append(f"- {eid}: {desc}")
        else:
            lines.append(f"- {eid}")
    return "\n".join(lines)


def _build_claim_chain(llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLAIM_EXTRACTION_SYSTEM),
            ("human", CLAIM_EXTRACTION_HUMAN),
        ]
    )
    return prompt | llm | StrOutputParser()


def extract_claims_from_text(
    chain,
    chunk_text: str,
    entities: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """Run claim extraction for one chunk. Returns [] on LLM or parse failure."""
    if not chunk_text or not str(chunk_text).strip():
        return []
    try:
        raw = chain.invoke(
            {
                "chunk_text": chunk_text,
                "entities": _format_entities_for_prompt(entities or []),
            }
        )
        return parse_claims_response(raw)
    except Exception as exc:
        logger.warning("Claim extraction LLM call failed (non-fatal): %s", exc)
        return []


def _fetch_chunks_with_entities(graph) -> List[Dict[str, Any]]:
    rows = graph.query(FETCH_CHUNKS_WITH_ENTITIES)
    return [dict(row) for row in rows if row.get("text")]


def extract_claims(
    graph,
    model: Optional[str] = None,
    email: Optional[str] = None,
    max_workers: int = MAX_CLAIM_WORKERS,
    chain=None,
) -> Dict[str, Any]:
    """
    Optional post-processing entrypoint: extract claims for all chunks with entities.

    Failures are logged; partial progress is persisted. Does not raise on per-chunk LLM errors.
    Pass `chain` to inject a prebuilt LLM chain (used by unit tests).
    """
    del email  # reserved for future token tracking parity with communities/raptor

    if model is None:
        from src.shared.common_fn import get_value_from_env

        model = get_value_from_env(
            "CLAIM_EXTRACTION_MODEL", CLAIM_EXTRACTION_DEFAULT_MODEL
        )
    chunks = _fetch_chunks_with_entities(graph)
    if not chunks:
        logger.info("extract_claims: no chunks with entities found")
        return {"chunks_processed": 0, "claims_extracted": 0, "claims_persisted": 0}

    model_name = model
    if chain is None:
        try:
            from src.llm import get_llm

            llm, model_name, _callback = get_llm(model)
            chain = _build_claim_chain(llm)
        except Exception as exc:
            logger.error("extract_claims: failed to initialize LLM (aborting task): %s", exc)
            return {
                "chunks_processed": 0,
                "claims_extracted": 0,
                "claims_persisted": 0,
                "error": str(exc),
            }

    logger.info(
        "extract_claims: processing %s chunks with model %s",
        len(chunks),
        model_name,
    )

    all_rows: List[Dict[str, Any]] = []
    extracted_count = 0

    def _process_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        entities = chunk.get("entities") or []
        claims = extract_claims_from_text(chain, chunk.get("text") or "", entities)
        if not claims:
            return []
        return claims_to_persist_rows(
            claims,
            chunk_element_id=chunk.get("element_id"),
            entities=entities,
        )

    workers = max(1, min(max_workers, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            try:
                rows = future.result()
            except Exception as exc:
                logger.warning("extract_claims: chunk worker failed (non-fatal): %s", exc)
                continue
            if rows:
                extracted_count += len(rows)
                all_rows.extend(rows)

    deduped: Dict[str, Dict[str, Any]] = {}
    for row in all_rows:
        existing = deduped.get(row["id"])
        if existing is None:
            deduped[row["id"]] = row
            continue
        merged_ids = list(
            dict.fromkeys((existing.get("entity_ids") or []) + (row.get("entity_ids") or []))
        )
        existing["entity_ids"] = merged_ids
        if not existing.get("chunk_element_id") and row.get("chunk_element_id"):
            existing["chunk_element_id"] = row["chunk_element_id"]

    persist_rows = list(deduped.values())
    persisted = 0
    try:
        persisted = persist_claims(graph, persist_rows)
    except Exception as exc:
        logger.error("extract_claims: persistence failed: %s", exc)
        return {
            "chunks_processed": len(chunks),
            "claims_extracted": extracted_count,
            "claims_persisted": 0,
            "error": str(exc),
        }

    logger.info(
        "extract_claims: extracted=%s persisted=%s chunks=%s",
        extracted_count,
        persisted,
        len(chunks),
    )
    return {
        "chunks_processed": len(chunks),
        "claims_extracted": extracted_count,
        "claims_persisted": persisted,
        "model": model_name,
    }
