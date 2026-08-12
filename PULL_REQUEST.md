# Full-Scale GraphRAG + RAPTOR Implementation

This pull request turns the Local-to-Global GraphRAG and RAPTOR POC designs in `POC_Documents/V1` into production capabilities inside the existing llm-relationship-graph-builder stack.

## What was built

### GraphRAG (Local-to-Global)
- Map-reduce global query-focused summarization over community summaries
- Explicit paper community levels **C0–C3** as chat modes (`global_c0` … `global_c3`) plus `global_map_reduce`
- Helpfulness scoring (0–100) on mapped partial answers, with score-ordered reduce
- Paper↔Neo4j level mapping (`paper_level` / `paper_level_label` on `__Community__`)
- Degree-prioritized leaf community summarization (GraphRAG packing)
- Optional **claim/covariate extraction** (`extract_claims`) → `__Claim__` nodes via `HAS_CLAIM`, included in leaf community summaries
- Bug fix: `/post_processing` `enable_communities` now passes LLM model and embedding args in the correct order

### RAPTOR
- Recursive GMM (+ UMAP/PCA) clustering and abstractive summarization over `Chunk` nodes
- Persisted as `__RaptorNode__` hierarchy with `HAS_CHILD` / `FROM_CHUNK` and vector index
- Pure ranking helpers for cosine / collapsed-tree selection
- Query modes: `raptor_collapsed` (collapsed tree) and `raptor_tree` (layer traversal)
- New post-processing job: `enable_raptor`
- Hardened empty-embedding / batching / missing-id guards

### Frontend
- New chat modes for GraphRAG map-reduce levels and RAPTOR strategies
- Mode gating: community modes require GDS + `enable_communities`; RAPTOR modes require `enable_raptor`
- Chat info modal surfaces GraphRAG level metadata and RAPTOR retrieved nodes (layer/score)
- Post-processing jobs: `extract_claims`, `enable_raptor`

## How to use
1. Extract documents as usual
2. Run post-processing with `extract_claims` (optional), `enable_communities`, and optionally `enable_raptor`
3. Deselect documents in the table
4. Choose a chat mode (`global map-reduce`, `global C0`–`C3`, `raptor collapsed`, or `raptor tree`)

## Config
See `backend/example.env` for GraphRAG/RAPTOR/claim env vars (`GRAPHRAG_*`, `RAPTOR_*`, `CLAIM_EXTRACTION_MODEL`, `COMMUNITY_CREATION_MODEL`, `RAPTOR_CREATION_MODEL`).

## Tests
- `backend/test_graphrag_raptor.py` — helpers, clustering, ranking, score wiring (19 passed, 1 skipped without full LLM deps)
- `backend/test_graphrag_claims.py` — claim parse/persist helpers (10 passed)
