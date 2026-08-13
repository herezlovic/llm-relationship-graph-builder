"""Unit tests for GraphRAG index-fidelity helpers: element clustering + parent packing + TS mode."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


class ElementSummaryClusteringTests(unittest.TestCase):
    def test_cluster_homogeneous_identical_vectors(self):
        from src.graphrag.element_summaries import cluster_homogeneous_indices

        embeddings = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        clusters = cluster_homogeneous_indices(embeddings, threshold=0.99)
        self.assertEqual(len(clusters), 2)
        sizes = sorted(len(c) for c in clusters)
        self.assertEqual(sizes, [1, 2])

    def test_cluster_homogeneous_empty_and_single(self):
        from src.graphrag.element_summaries import cluster_homogeneous_indices

        self.assertEqual(cluster_homogeneous_indices([]), [])
        self.assertEqual(cluster_homogeneous_indices([[0.1, 0.2]]), [[0]])

    def test_cluster_respects_max_cluster_size(self):
        from src.graphrag.element_summaries import cluster_homogeneous_indices

        # All identical → one component, then partitioned
        embeddings = [[1.0, 0.0] for _ in range(5)]
        clusters = cluster_homogeneous_indices(embeddings, threshold=0.9, max_cluster_size=2)
        self.assertTrue(all(len(c) <= 2 for c in clusters))
        self.assertEqual(sum(len(c) for c in clusters), 5)

    def test_select_description_prefers_element_summary(self):
        from src.graphrag.element_summaries import select_description_for_prompt

        self.assertEqual(
            select_description_for_prompt(
                {"description": "raw", "element_summary": "consolidated"}
            ),
            "consolidated",
        )
        self.assertEqual(select_description_for_prompt({"description": "raw"}), "raw")


class HigherLevelPackingTests(unittest.TestCase):
    def test_no_substitution_when_under_budget(self):
        from src.graphrag.helpers import build_higher_level_community_context

        packed, meta = build_higher_level_community_context(
            [
                {"id": "a", "summary": "short a", "element_text": "element a text"},
                {"id": "b", "summary": "short b", "element_text": "element b text"},
            ],
            max_tokens=500,
        )
        self.assertEqual(meta["substitutions"], 0)
        self.assertIn("element a text", packed)
        self.assertIn("element b text", packed)

    def test_substitutes_largest_element_block_first(self):
        from src.graphrag.helpers import build_higher_level_community_context

        packed, meta = build_higher_level_community_context(
            [
                {
                    "id": "big",
                    "summary": "BIG_SUMMARY",
                    "element_text": "X" * 400,  # ~100 tokens
                },
                {
                    "id": "small",
                    "summary": "SMALL_SUMMARY",
                    "element_text": "Y" * 40,  # ~10 tokens
                },
            ],
            max_tokens=30,
        )
        self.assertGreaterEqual(meta["substitutions"], 1)
        self.assertIn("BIG_SUMMARY", packed)
        # Small elements may still be present if budget allows after big substitution
        self.assertTrue("Y" in packed or "SMALL_SUMMARY" in packed)

    def test_falls_back_to_summaries_without_elements(self):
        from src.graphrag.helpers import build_higher_level_community_context

        packed, meta = build_higher_level_community_context(
            [
                {"id": "a", "summary": "only summary a"},
                {"id": "b", "summary": "only summary b"},
            ],
            max_tokens=10,
        )
        self.assertEqual(meta["substitutions"], 0)
        self.assertIn("only summary a", packed)
        self.assertIn("only summary b", packed)

    def test_prepare_community_string_uses_element_summary(self):
        from src.graphrag.helpers import prepare_community_string

        text = prepare_community_string(
            {
                "nodes": [
                    {
                        "id": "Acme",
                        "type": "Org",
                        "description": "raw desc",
                        "element_summary": "consolidated Acme summary",
                        "degree": 3,
                    }
                ],
                "rels": [],
            }
        )
        self.assertIn("consolidated Acme summary", text)
        self.assertNotIn("raw desc", text)


class TSMapReduceModeTests(unittest.TestCase):
    def test_ts_mode_registered(self):
        gs_path = os.path.join(os.path.dirname(__file__), "src", "graphrag", "global_search.py")
        with open(gs_path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('CHAT_TS_MAP_REDUCE_MODE = "ts_map_reduce"', source)
        self.assertIn("CHAT_TS_MAP_REDUCE_MODE", source)
        self.assertIn("GET_SOURCE_CHUNK_TEXTS", source)
        self.assertIn('source_key in {"chunks", "ts", "text", "source_texts"}', source)

    def test_run_global_map_reduce_ts_source(self):
        try:
            from src.graphrag import global_search as gs
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional backend deps missing: {exc}")

        class FakeLLM:
            def invoke(self, *_args, **_kwargs):
                return self

            def __or__(self, _other):
                return self

        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = [
            "helpfulness: 80\nanswer: Partial from source text",
            "Final TS global answer",
        ]

        with patch.object(gs, "get_llm", return_value=(FakeLLM(), "test-model", None)), patch.object(
            gs, "_build_map_chain", return_value=fake_chain
        ), patch.object(gs, "_build_reduce_chain", return_value=fake_chain), patch.object(
            gs, "get_value_from_env", side_effect=lambda *a, **k: a[1] if len(a) > 1 else None
        ):
            graph = MagicMock()
            graph.query.return_value = [
                {
                    "id": "c1",
                    "community_id": "chunk-1",
                    "title": "doc.pdf",
                    "summary": "Source text about sports " * 30,
                    "level": 0,
                }
            ]
            result = gs.run_global_map_reduce(
                graph, "openai_gpt_5_mini", "What themes?", source="ts"
            )
            self.assertEqual(result["source"], "ts")
            self.assertIn("ts global answer", result["answer"].lower())
            # TS path must not query community max level
            called_queries = [c.args[0] for c in graph.query.call_args_list if c.args]
            self.assertTrue(any("Chunk" in q for q in called_queries))


class ScoreElementSummaryWiringTests(unittest.TestCase):
    def test_consolidate_element_summaries_wired_in_score(self):
        score_path = os.path.join(os.path.dirname(__file__), "score.py")
        with open(score_path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('if "consolidate_element_summaries" in tasks:', source)
        self.assertIn("consolidate_element_summaries,", source)
        self.assertIn('get_value_from_env(\n                "ELEMENT_SUMMARY_MODEL"', source)
        self.assertIn('if "extract_claims" in tasks:', source)


class CommunitiesParentQueryTests(unittest.TestCase):
    def test_parent_level_query_present(self):
        communities_path = os.path.join(os.path.dirname(__file__), "src", "communities.py")
        with open(communities_path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("GET_PARENT_COMMUNITY_INFO_AT_LEVEL", source)
        self.assertIn("$level", source)
        self.assertIn("element_summary", source)
        self.assertIn("GET_LEAF_ELEMENTS_UNDER_PARENT", source)
        self.assertIn("build_higher_level_community_context", source)


if __name__ == "__main__":
    unittest.main()
