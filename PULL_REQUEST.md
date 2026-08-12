# Full-Scale GraphRAG + RAPTOR Implementation

This pull request turns the Local-to-Global GraphRAG and RAPTOR POC designs in `POC_Documents/V1` into production capabilities inside the existing llm-relationship-graph-builder stack.

## What was built

### GraphRAG (Local-to-Global)
- Map-reduce global query-focused summarization over community summaries
- Explicit paper community levels **C0–C3** as chat modes (`global_c0` … `global_c3`) plus `global_map_reduce`
- Helpfulness scoring (0–100) on mapped partial answers, with score-ordered reduce
- Paper↔Neo4j level mapping (`paper_level` / `paper_level_label` on `__Community__`)
- Degree-prioritized leaf community summarization (GraphRAG packing)
- Bugfix: `/post_processing` `enable_communities` now passes LLM model and embedding args in the correct order

### RAPTOR
- Recursive GMM (+ UMAP/PCA) clustering and abstractive summarization over `Chunk` nodes
- Persisted as `__RaptorNode__` hierarchy with `HAS_CHILD` / `FROM_CHUNK` and vector index
- Query modes: `raptor_collapsed` (collapsed tree) and `raptor_tree` (layer traversal)
- New post-processing job: `enable_raptor`

### Frontend
- New chat modes for GraphRAG map-reduce levels and RAPTOR strategies
- Mode gating: community modes require GDS + `enable_communities`; RAPTOR modes require `enable_raptor`

## How to use
1. Extract documents as usual
2. Run post-processing with `enable_communities` (and optionally `enable_raptor`)
3. Deselect documents in the table
4. Choose a new chat mode (`global map-reduce`, `global C0`–`C3`, `raptor collapsed`, or `raptor tree`)

## Tests
- `backend/test_graphrag_raptor.py` covers level mapping, chunking, map parsing, clustering, prepare_string prioritization, and a mocked map-reduce flow
