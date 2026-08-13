# GraphRAG Index Fidelity + TS Map-Reduce

Continues the full-scale GraphRAG/RAPTOR work by closing the largest remaining
Local-to-Global index-construction gaps and adding the paper's TS comparison mode.

## What was built

### Higher-level community summarization (token-budget substitution)
- Parents are summarized **bottom-up, one Leiden level at a time**
- Level-1 parents pack leaf **element texts** first; when over budget, the largest
  sub-community element blocks are **substituted** with that sub-community's summary
  (GraphRAG paper packing)
- Env: `GRAPHRAG_PARENT_SUMMARY_TOKEN_BUDGET`

### Element-summary homogeneous clustering
- New post-processing job: `consolidate_element_summaries`
- Embeds entity descriptions → cosine connected-component clusters → LLM consolidates
  multi-member clusters into `element_summary`
- Leaf community prompts prefer `element_summary` over raw `description`
- Env: `ELEMENT_SUMMARY_MODEL`, `GRAPHRAG_ELEMENT_SIMILARITY_THRESHOLD`,
  `GRAPHRAG_ELEMENT_MAX_CLUSTER_SIZE`, `GRAPHRAG_ELEMENT_SUMMARY_WORKERS`

### Paper TS map-reduce chat mode
- New mode `ts_map_reduce` — same map-reduce/helpfulness/reduce path over **Chunk**
  source texts (graph-free baseline from the Local-to-Global paper)
- Always available in the UI (does not require GDS / `enable_communities`)
- Chat info shows "TS (source texts)" metadata
- Env: `GRAPHRAG_TS_CHUNK_LIMIT`
- Note: paper **SS** remains the existing `vector` semantic-search mode

### Docs / wiring
- Documented `CLAIM_EXTRACTION_MODEL` and new GraphRAG env vars in `backend/example.env`
- Frontend post-processing checklist includes `consolidate_element_summaries`

## Suggested post-processing order
1. `extract_claims` (optional)
2. `consolidate_element_summaries` (optional, before communities)
3. `enable_communities`
4. `enable_raptor` (optional)

## Tests
- `backend/test_graphrag_index_fidelity.py` — clustering, packing, TS mode, wiring
- Existing `test_graphrag_raptor.py` / `test_graphrag_claims.py` remain
