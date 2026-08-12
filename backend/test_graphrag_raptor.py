"""Unit tests for full-scale GraphRAG map-reduce helpers and RAPTOR clustering/ranking."""

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

    def test_paper_level_clamping(self):
        from src.graphrag.helpers import resolve_paper_community_level

        # Paper levels outside C0–C3 are clamped before mapping.
        self.assertEqual(resolve_paper_community_level(-2, 3), 3)  # clamps to C0 → neo max
        self.assertEqual(resolve_paper_community_level(99, 3), 0)  # clamps to C3 → neo 0
        self.assertEqual(resolve_paper_community_level(1, 0), 0)  # neo level never negative
        self.assertEqual(resolve_paper_community_level(0, -1), 0)  # negative max → 0
        self.assertEqual(resolve_paper_community_level("2", 4), 2)

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

    def test_parse_map_response_edge_cases(self):
        from src.graphrag.helpers import parse_map_response

        # Missing helpfulness → 0; whole raw text kept as answer when no answer: prefix
        helpfulness, answer = parse_map_response("Just some free-form map output.")
        self.assertEqual(helpfulness, 0)
        self.assertIn("free-form", answer)

        # Clamp >100 and case-insensitive keys
        helpfulness, answer = parse_map_response("Helpfulness: 250\nAnswer: Over-scored")
        self.assertEqual(helpfulness, 100)
        self.assertEqual(answer, "Over-scored")

        # Empty / whitespace
        helpfulness, answer = parse_map_response("   ")
        self.assertEqual(helpfulness, 0)
        self.assertEqual(answer, "")

        # Multiline answer body
        helpfulness, answer = parse_map_response("helpfulness: 40\nanswer: Line one\nLine two")
        self.assertEqual(helpfulness, 40)
        self.assertIn("Line two", answer)

        # helpfulness without answer key still extracts score
        helpfulness, answer = parse_map_response("helpfulness: 7\nno answer prefix here")
        self.assertEqual(helpfulness, 7)
        self.assertIn("helpfulness", answer.lower())


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

    def test_cluster_embeddings_empty(self):
        from src.raptor.clustering import cluster_embeddings

        self.assertEqual(cluster_embeddings([]), [])
        self.assertEqual(cluster_embeddings([], max_clusters=5), [])

    def test_cluster_embeddings_single_vector(self):
        from src.raptor.clustering import cluster_embeddings

        self.assertEqual(cluster_embeddings([[0.1, 0.2, 0.3]]), [[0]])
        self.assertEqual(cluster_embeddings([[1.0]]), [[0]])

    def test_cluster_embeddings_two_vectors(self):
        from src.raptor.clustering import cluster_embeddings

        self.assertEqual(cluster_embeddings([[1.0, 0.0], [0.0, 1.0]]), [[0, 1]])

    def test_cluster_embeddings_soft_membership(self):
        """Points with membership probability >= threshold may appear in multiple clusters."""
        from src.raptor import clustering as clustering_mod
        import numpy as np

        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.6, 0.4],
        ]
        fake_probs = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.55, 0.45],  # soft: both >= 0.4
                [0.7, 0.3],
            ]
        )

        class FakeGMM:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, _x):
                return self

            def predict_proba(self, _x):
                return fake_probs

        with patch.object(clustering_mod, "_reduce_dimensions", side_effect=lambda x, **k: x), patch.object(
            clustering_mod, "_bic_select_components", return_value=2
        ), patch("sklearn.mixture.GaussianMixture", FakeGMM):
            clusters = clustering_mod.cluster_embeddings(embeddings, max_clusters=2, threshold=0.4)

        self.assertEqual(len(clusters), 2)
        # Soft member (index 2) belongs to both clusters
        self.assertIn(2, clusters[0])
        self.assertIn(2, clusters[1])
        # Hard members stay in one cluster
        self.assertIn(0, clusters[0])
        self.assertNotIn(0, clusters[1])
        self.assertIn(1, clusters[1])
        self.assertNotIn(1, clusters[0])

    def test_recursive_cluster_splits_on_token_budget(self):
        from src.raptor.clustering import recursive_cluster_indices

        embeddings = [[float(i), 0.0, 0.0] for i in range(6)]
        token_counts = [1000] * 6
        clusters = recursive_cluster_indices(embeddings, token_counts, max_tokens=1500, max_clusters=3)
        self.assertTrue(all(sum(token_counts[i] for i in c) <= 1500 or len(c) == 1 for c in clusters))

    def test_recursive_cluster_empty(self):
        from src.raptor.clustering import recursive_cluster_indices

        self.assertEqual(recursive_cluster_indices([], [], max_tokens=100), [])


class RaptorRankingHelperTests(unittest.TestCase):
    """Pure ranking/selection tests — no LLM / Neo4j stack required."""

    def test_cosine_similarity(self):
        from src.raptor.ranking import cosine_similarity

        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0, places=5)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=5)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0, places=5)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)
        self.assertEqual(cosine_similarity([], [1, 0]), 0.0)
        self.assertEqual(cosine_similarity([1, 0], [1, 0, 0]), 0.0)  # shape mismatch

    def test_rank_nodes_skips_missing_embeddings(self):
        from src.raptor.ranking import rank_nodes

        nodes = [
            {"id": "a", "embedding": [1.0, 0.0], "text": "alpha"},
            {"id": "b", "embedding": None, "text": "beta"},
            {"id": "c", "embedding": [], "text": "gamma"},
            {"id": "d", "embedding": [0.0, 1.0], "text": "delta"},
        ]
        ranked = rank_nodes([1.0, 0.0], nodes)
        self.assertEqual([n["id"] for n in ranked], ["a", "d"])
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])

    def test_select_collapsed_nodes_respects_top_k_and_budget(self):
        from src.raptor.ranking import select_collapsed_nodes

        ranked = [
            {"id": "1", "text": "a" * 40, "score": 0.9},  # ~10 tokens
            {"id": "2", "text": "b" * 40, "score": 0.8},
            {"id": "3", "text": "c" * 40, "score": 0.7},
            {"id": "4", "text": "d" * 4, "score": 0.6},  # ~1 token
        ]
        # Ample budget → hard-cap at top_k in score order
        selected = select_collapsed_nodes(ranked, top_k=2, max_context_tokens=100)
        self.assertEqual([n["id"] for n in selected], ["1", "2"])

        # Tight budget → skip oversized mid-ranked nodes and pack a smaller one
        packed = select_collapsed_nodes(ranked, top_k=3, max_context_tokens=12)
        self.assertEqual([n["id"] for n in packed], ["1", "4"])

    def test_select_collapsed_nodes_empty(self):
        from src.raptor.ranking import select_collapsed_nodes

        self.assertEqual(select_collapsed_nodes([], top_k=5, max_context_tokens=100), [])

    def test_collapsed_retrieve_with_mocked_graph(self):
        """Collapsed selection via ranking helpers + mocked graph rows (no LLM)."""
        from src.raptor.ranking import rank_nodes, select_collapsed_nodes

        graph_rows = [
            {"id": "n1", "text": "sports " * 20, "embedding": [1.0, 0.0], "layer": 0, "file_names": ["a"]},
            {"id": "n2", "text": "business " * 20, "embedding": [0.0, 1.0], "layer": 1, "file_names": ["b"]},
            {"id": "n3", "text": "mixed", "embedding": None, "layer": 0, "file_names": []},
        ]
        ranked = rank_nodes([0.9, 0.1], graph_rows)
        selected = select_collapsed_nodes(ranked, top_k=1, max_context_tokens=500)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "n1")


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


class ScorePostProcessingWiringTests(unittest.TestCase):
    def test_enable_raptor_and_communities_wiring_in_score(self):
        """score.py wires enable_communities / enable_raptor to the expected call shapes."""
        score_path = os.path.join(os.path.dirname(__file__), "score.py")
        with open(score_path, encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn('if "enable_communities" in tasks:', source)
        self.assertIn('get_value_from_env(\n                "COMMUNITY_CREATION_MODEL"', source)
        self.assertIn("create_communities,", source)

        self.assertIn('if "enable_raptor" in tasks:', source)
        self.assertIn('get_value_from_env(\n                "RAPTOR_CREATION_MODEL"', source)
        self.assertIn("create_raptor_index,", source)
        # Positional args: graph, model, embedding_provider, embedding_model, email
        self.assertIn("credentials.email,", source)
        self.assertRegex(
            source,
            r"create_raptor_index,\s*graph,\s*raptor_model,\s*embedding_provider,\s*embedding_model,\s*credentials\.email,",
        )

        # tree_builder signature without importing LLM-heavy modules
        tb_path = os.path.join(os.path.dirname(__file__), "src", "raptor", "tree_builder.py")
        with open(tb_path, encoding="utf-8") as handle:
            tb = handle.read()
        self.assertIn(
            "def create_raptor_index(\n    graph,\n    model: str,\n    embedding_provider: Optional[str] = None,\n"
            "    embedding_model: Optional[str] = None,\n    email: Optional[str] = None,\n)",
            tb,
        )


if __name__ == "__main__":
    unittest.main()
