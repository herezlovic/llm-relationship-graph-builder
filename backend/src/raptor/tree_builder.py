"""Build and persist a RAPTOR tree over document chunks in Neo4j."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.raptor.clustering import recursive_cluster_indices
from src.raptor.prompts import RAPTOR_SUMMARY_HUMAN, RAPTOR_SUMMARY_SYSTEM
from src.shared.common_fn import get_value_from_env, load_embedding_model

logger = logging.getLogger(__name__)

RAPTOR_NODE_LABEL = "__RaptorNode__"
CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_SUMMARY_TOKEN_LIMIT = 3500
DEFAULT_MAX_CLUSTERS = 10
DEFAULT_MAX_LAYERS = 5

DROP_RAPTOR_NODES = f"MATCH (n:{RAPTOR_NODE_LABEL}) DETACH DELETE n"

CREATE_RAPTOR_CONSTRAINT = (
    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{RAPTOR_NODE_LABEL}) REQUIRE n.id IS UNIQUE"
)

FETCH_CHUNKS_QUERY = """
MATCH (c:Chunk)-[:PART_OF]->(d:Document)
WHERE c.text IS NOT NULL
RETURN elementId(c) AS element_id,
       c.id AS chunk_id,
       c.text AS text,
       d.fileName AS file_name
ORDER BY d.fileName, c.position
"""

STORE_RAPTOR_NODES = """
UNWIND $rows AS row
MERGE (n:__RaptorNode__ {id: row.id})
SET n.text = row.text,
    n.layer = row.layer,
    n.is_leaf = row.is_leaf,
    n.token_count = row.token_count,
    n.source_chunk_ids = row.source_chunk_ids,
    n.file_names = row.file_names
WITH n, row
CALL db.create.setNodeVectorProperty(n, "embedding", row.embedding)
RETURN count(n) AS stored
"""

LINK_RAPTOR_CHILDREN = """
UNWIND $rows AS row
MATCH (parent:__RaptorNode__ {id: row.parent_id})
MATCH (child:__RaptorNode__ {id: row.child_id})
MERGE (parent)-[:HAS_CHILD]->(child)
"""

LINK_RAPTOR_TO_CHUNK = """
UNWIND $rows AS row
MATCH (leaf:__RaptorNode__ {id: row.leaf_id})
MATCH (c:Chunk) WHERE elementId(c) = row.chunk_element_id
MERGE (leaf)-[:FROM_CHUNK]->(c)
"""

CREATE_RAPTOR_VECTOR_INDEX = """
CREATE VECTOR INDEX raptor_vector IF NOT EXISTS FOR (n:__RaptorNode__) ON n.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: $dimension,
    `vector.similarity_function`: 'cosine'
  }
}
"""

DROP_RAPTOR_VECTOR_INDEX = "DROP INDEX raptor_vector IF EXISTS"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN_ESTIMATE)


def _summarize_cluster(chain, passages: Sequence[str]) -> str:
    joined = "\n\n----\n\n".join(passages)
    return chain.invoke({"passages": joined}).strip()


def _build_summary_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAPTOR_SUMMARY_SYSTEM),
            ("human", RAPTOR_SUMMARY_HUMAN),
        ]
    )
    return prompt | llm | StrOutputParser()


def clear_raptor_index(graph) -> None:
    graph.query(DROP_RAPTOR_NODES)
    try:
        graph.query(DROP_RAPTOR_VECTOR_INDEX)
    except Exception as exc:
        logger.debug("Could not drop RAPTOR vector index: %s", exc)


def _fetch_chunks(graph) -> List[Dict[str, Any]]:
    rows = graph.query(FETCH_CHUNKS_QUERY)
    return [dict(row) for row in rows if row.get("text")]


def create_raptor_index(
    graph,
    model: str,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
    email: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a RAPTOR tree from Chunk nodes and persist it as __RaptorNode__ hierarchy.
    """
    logger.info("Starting RAPTOR index creation.")
    graph.query(CREATE_RAPTOR_CONSTRAINT)
    clear_raptor_index(graph)

    chunks = _fetch_chunks(graph)
    if not chunks:
        logger.warning("No chunks available for RAPTOR indexing.")
        return {"layers": 0, "nodes": 0, "message": "No chunks found"}

    embeddings_model, dimension = load_embedding_model(embedding_provider, embedding_model)
    llm, model_name, _ = get_llm(model)
    summary_chain = _build_summary_chain(llm)

    max_tokens = get_value_from_env("RAPTOR_SUMMARY_TOKEN_LIMIT", DEFAULT_SUMMARY_TOKEN_LIMIT, "int")
    max_clusters = get_value_from_env("RAPTOR_MAX_CLUSTERS", DEFAULT_MAX_CLUSTERS, "int")
    max_layers = get_value_from_env("RAPTOR_MAX_LAYERS", DEFAULT_MAX_LAYERS, "int")
    workers = get_value_from_env("RAPTOR_SUMMARY_WORKERS", 6, "int")

    # Layer 0 = leaf nodes from chunks
    current_nodes: List[Dict[str, Any]] = []
    leaf_chunk_links: List[Dict[str, str]] = []
    for chunk in chunks:
        node_id = str(uuid.uuid4())
        text = chunk["text"]
        embedding = embeddings_model.embed_query(text)
        current_nodes.append(
            {
                "id": node_id,
                "text": text,
                "layer": 0,
                "is_leaf": True,
                "token_count": _estimate_tokens(text),
                "source_chunk_ids": [chunk.get("chunk_id")],
                "file_names": [chunk.get("file_name")],
                "embedding": embedding,
            }
        )
        leaf_chunk_links.append(
            {"leaf_id": node_id, "chunk_element_id": chunk["element_id"]}
        )

    all_nodes = list(current_nodes)
    parent_child_links: List[Dict[str, str]] = []

    layer = 0
    while layer < max_layers and len(current_nodes) > 1:
        layer += 1
        vectors = [n["embedding"] for n in current_nodes]
        token_counts = [n["token_count"] for n in current_nodes]
        clusters = recursive_cluster_indices(
            vectors,
            token_counts,
            max_tokens=max_tokens,
            max_clusters=max_clusters,
        )

        # Stop if clustering no longer compresses the layer.
        if len(clusters) >= len(current_nodes):
            logger.info("RAPTOR clustering no longer compresses at layer %s; stopping.", layer)
            break

        multi_clusters = [c for c in clusters if len(c) > 1]
        if not multi_clusters:
            logger.info("No multi-member RAPTOR clusters at layer %s; stopping.", layer)
            break

        next_nodes: List[Dict[str, Any]] = []

        def _build_parent(member_indices: List[int]) -> Optional[Dict[str, Any]]:
            members = [current_nodes[i] for i in member_indices]
            passages = [m["text"] for m in members]
            try:
                summary = _summarize_cluster(summary_chain, passages)
            except Exception as exc:
                logger.error("RAPTOR summary failed: %s", exc)
                summary = "\n".join(passages)[: max_tokens * CHARS_PER_TOKEN_ESTIMATE]

            embedding = embeddings_model.embed_query(summary)
            parent_id = str(uuid.uuid4())
            source_chunk_ids = []
            file_names = []
            for member in members:
                source_chunk_ids.extend(member.get("source_chunk_ids") or [])
                file_names.extend(member.get("file_names") or [])
            node = {
                "id": parent_id,
                "text": summary,
                "layer": layer,
                "is_leaf": False,
                "token_count": _estimate_tokens(summary),
                "source_chunk_ids": sorted({c for c in source_chunk_ids if c}),
                "file_names": sorted({f for f in file_names if f}),
                "embedding": embedding,
            }
            return {"node": node, "children": [m["id"] for m in members]}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(_build_parent, cluster) for cluster in multi_clusters]
            for future in as_completed(futures):
                built = future.result()
                if not built:
                    continue
                node = built["node"]
                next_nodes.append(node)
                all_nodes.append(node)
                for child_id in built["children"]:
                    parent_child_links.append({"parent_id": node["id"], "child_id": child_id})

        # Unclustered / singleton nodes remain addressable via collapsed search;
        # only summarized parents form the next hierarchical layer.
        current_nodes = next_nodes
        if len(current_nodes) <= 1:
            break

    # Persist nodes in batches
    batch_size = 50
    for i in range(0, len(all_nodes), batch_size):
        batch = all_nodes[i : i + batch_size]
        graph.query(STORE_RAPTOR_NODES, params={"rows": batch})

    if parent_child_links:
        for i in range(0, len(parent_child_links), batch_size):
            graph.query(LINK_RAPTOR_CHILDREN, params={"rows": parent_child_links[i : i + batch_size]})

    if leaf_chunk_links:
        for i in range(0, len(leaf_chunk_links), batch_size):
            graph.query(LINK_RAPTOR_TO_CHUNK, params={"rows": leaf_chunk_links[i : i + batch_size]})

    try:
        graph.query(DROP_RAPTOR_VECTOR_INDEX)
    except Exception:
        pass
    try:
        graph.query(CREATE_RAPTOR_VECTOR_INDEX, params={"dimension": int(dimension)})
    except Exception as exc:
        logger.warning("Could not create RAPTOR vector index: %s", exc)

    max_layer = max((n["layer"] for n in all_nodes), default=0)
    result = {
        "layers": max_layer + 1,
        "nodes": len(all_nodes),
        "leaves": sum(1 for n in all_nodes if n.get("is_leaf")),
        "model": model_name,
        "embedding_dimension": dimension,
        "message": "RAPTOR index created successfully",
    }
    logger.info("RAPTOR index created: %s", result)
    return result
