"""Unit tests for full-scale GraphRAG map-reduce helpers and RAPTOR clustering."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


class GraphRAGHelperTests(unittest.TestCase):
    def test_paper_level_mapping(self):
        from src.graphrag.helpers import resolve_paper_community_level

        self.assertEqual(resolve_paper_community_level(0, 3), 3)
        self.assertEqual(resolve_paper_community_level(1, 3), 2)
        self.assertEqual(resolve_paper_community_level(2, 3), 1)
        self.assertEqual(resolve_paper_community_level(3, 3), 0)
        self.assertIsNone(resolve_paper_community_level(None, 3))
        self.assertEqual(resolve_paper_community_level(0, 1), 1)
        self.assertEqual(resolve_paper_community_level(3, 1), 0)

    def test_chunk_summaries_respect_budget(self):
        from src.graphrag.helpers import chunk_summaries

        summaries = [{"summary": "a" * 40, "id": str(i)} for i in range(10)]
        chunks = chunk_summaries(summaries, max_tokens=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(sum(len(c) for c in chunks), 10)

    def test_parse_map_response(self):
        from src.graphrag.helpers import parse_map_response

        helpfulness, answer = parse_map_response("helpfulness: 82\nanswer: Sports themes dominate.")
        self.assertEqual(helpfulness, 82)
        self.assertIn("Sports", answer)

        helpfulness, answer = parse_map_response("helpfulness: 0\nanswer: Not useful")
        self.assertEqual(helpfulness, 0)


class RaptorClusteringTests(unittest.TestCase):
    def test_cluster_embeddings_small(self):
        from src.raptor.clustering import cluster_embeddings

        embeddings = [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
        ]
        clusters = cluster_embeddings(embeddings, max_clusters=2, threshold=0.2)
        self.assertGreaterEqual(len(clusters), 1)
        covered = sorted({idx for cluster in clusters for idx in cluster})
        self.assertEqual(covered, [0, 1, 2, 3])

    def test_recursive_cluster_splits_on_token_budget(self):
        from src.raptor.clustering import recursive_cluster_indices

        embeddings = [[float(i), 0.0, 0.0] for i in range(6)]
        token_counts = [1000] * 6
        clusters = recursive_cluster_indices(embeddings, token_counts, max_tokens=1500, max_clusters=3)
        self.assertTrue(all(sum(token_counts[i] for i in c) <= 1500 or len(c) == 1 for c in clusters))


class CommunityPrepareStringTests(unittest.TestCase):
    def test_prepare_string_prioritizes_degree(self):
        from src.graphrag.helpers import prepare_community_string

        community = {
            "nodes": [
                {"id": "low", "type": "Person", "description": "a", "degree": 1},
                {"id": "high", "type": "Org", "description": "b", "degree": 9},
            ],
            "rels": [
                {"start": "low", "type": "KNOWS", "end": "high", "combined_degree": 10},
                {"start": "a", "type": "OTHER", "end": "b", "combined_degree": 1},
            ],
        }
        text = prepare_community_string(community, max_chars=500)
        self.assertLess(text.find("id: high"), text.find("id: low"))
        self.assertIn("KNOWS", text)


class GraphRAGMapReduceFlowTests(unittest.TestCase):
    def test_run_global_map_reduce_happy_path(self):
        try:
            from src.graphrag import global_search as gs
        except ModuleNotFoundError as exc:
            self.skipTest(f"Optional backend deps missing for map-reduce flow test: {exc}")

        class FakeLLM:
            def invoke(self, *_args, **_kwargs):
                return self

            def __or__(self, _other):
                return self

        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = [
            "helpfulness: 90\nanswer: Partial A about sports",
            "helpfulness: 70\nanswer: Partial B about business",
            "Final combined global answer",
        ]

        with patch.object(gs, "get_llm", return_value=(FakeLLM(), "test-model", None)), patch.object(
            gs, "_build_map_chain", return_value=fake_chain
        ), patch.object(gs, "_build_reduce_chain", return_value=fake_chain), patch.object(
            gs, "get_value_from_env", side_effect=lambda *a, **k: a[1] if len(a) > 1 else None
        ):
            graph = MagicMock()
            graph.query.side_effect = [
                [{"max_level": 2}],
                [
                    {
                        "id": "1",
                        "community_id": "2-1",
                        "title": "Sports",
                        "summary": "Sports summary " * 20,
                        "level": 2,
                    },
                    {
                        "id": "2",
                        "community_id": "2-2",
                        "title": "Business",
                        "summary": "Business summary " * 20,
                        "level": 2,
                    },
                ],
            ]
            result = gs.run_global_map_reduce(
                graph, "openai_gpt_5_mini", "What are the themes?", paper_level=0
            )
            self.assertIn("global answer", result["answer"].lower())
            self.assertEqual(result["model"], "test-model")
            self.assertEqual(result["neo4j_level"], 2)


if __name__ == "__main__":
    unittest.main()
