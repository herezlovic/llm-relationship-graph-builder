"""RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."""

def __getattr__(name):
    if name in {"RAPTOR_MODES", "process_raptor_response"}:
        from src.raptor import query as _query

        return getattr(_query, name)
    if name == "create_raptor_index":
        from src.raptor.tree_builder import create_raptor_index

        return create_raptor_index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RAPTOR_MODES",
    "create_raptor_index",
    "process_raptor_response",
]
