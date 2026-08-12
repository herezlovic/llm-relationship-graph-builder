"""Microsoft-style Local-to-Global GraphRAG map-reduce search."""

def __getattr__(name):
    if name in {
        "GRAPH_RAG_GLOBAL_MODES",
        "process_graphrag_global_response",
        "resolve_paper_community_level",
    }:
        from src.graphrag import global_search as _global_search

        return getattr(_global_search, name)
    if name in {"extract_claims", "CLAIM_EXTRACTION_DEFAULT_MODEL"}:
        from src.graphrag import claims as _claims

        return getattr(_claims, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GRAPH_RAG_GLOBAL_MODES",
    "process_graphrag_global_response",
    "resolve_paper_community_level",
    "extract_claims",
    "CLAIM_EXTRACTION_DEFAULT_MODEL",
]
