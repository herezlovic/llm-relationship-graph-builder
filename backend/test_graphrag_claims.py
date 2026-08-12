"""Unit tests for GraphRAG claim covariate extraction and persistence helpers."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


class ClaimParseTests(unittest.TestCase):
    def test_parse_claims_response_blocks(self):
        from src.graphrag.claims import parse_claims_response

        raw = """
===CLAIM===
subject: Acme Corp
object: Widget Inc
type: acquisition
description: Acme acquired Widget Inc
source_span: Acme Corp acquired Widget Inc in 2020
start_date: 2020-01-01
end_date:
status: known

===CLAIM===
subject: Jane Doe
object: Acme Corp
type: affiliation
description: Jane Doe is CEO of Acme Corp
source_span: Jane Doe, CEO of Acme Corp
start_date:
end_date:
status: known
"""
        claims = parse_claims_response(raw)
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["subject"], "Acme Corp")
        self.assertEqual(claims[0]["type"], "acquisition")
        self.assertEqual(claims[0]["source_span"], "Acme Corp acquired Widget Inc in 2020")
        self.assertEqual(claims[1]["subject"], "Jane Doe")
        self.assertEqual(claims[1]["object"], "Acme Corp")

    def test_parse_no_claims(self):
        from src.graphrag.claims import parse_claims_response

        self.assertEqual(parse_claims_response("NO_CLAIMS"), [])
        self.assertEqual(parse_claims_response(""), [])
        self.assertEqual(parse_claims_response(None), [])

    def test_make_claim_id_is_stable(self):
        from src.graphrag.claims import make_claim_id

        a = make_claim_id("Acme", "Widget", "acquisition", "Acme acquired Widget", "span")
        b = make_claim_id("acme", "widget", "Acquisition", "acme acquired widget", "span")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("claim-"))


class ClaimPersistHelperTests(unittest.TestCase):
    def test_claims_to_persist_rows_resolves_entities(self):
        from src.graphrag.claims import claims_to_persist_rows

        claims = [
            {
                "subject": "Acme Corp",
                "object": "Widget",
                "type": "acquisition",
                "description": "Acme acquired Widget",
                "source_span": "Acme Corp acquired Widget",
                "start_date": "2020-01-01",
                "end_date": "",
                "status": "known",
            }
        ]
        entities = [
            {"id": "Acme Corp", "description": "A company"},
            {"id": "Other", "description": "Unrelated"},
        ]
        rows = claims_to_persist_rows(
            claims, chunk_element_id="elem-1", entities=entities
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_element_id"], "elem-1")
        self.assertIn("Acme Corp", rows[0]["entity_ids"])
        self.assertEqual(rows[0]["start_date"], "2020-01-01")
        self.assertIsNone(rows[0]["end_date"])
        self.assertTrue(rows[0]["id"].startswith("claim-"))

    def test_persist_claims_runs_constraint_and_merge(self):
        from src.graphrag.claims import persist_claims

        graph = MagicMock()
        graph.query.side_effect = [None, [{"stored": 2}]]
        rows = [
            {
                "id": "claim-abc",
                "subject": "A",
                "object": "B",
                "type": "event",
                "description": "desc",
                "source_span": "span",
                "start_date": None,
                "end_date": None,
                "status": None,
                "chunk_element_id": "c1",
                "entity_ids": ["A"],
            }
        ]
        stored = persist_claims(graph, rows)
        self.assertEqual(stored, 2)
        self.assertEqual(graph.query.call_count, 2)
        # First call is constraint
        first_q = graph.query.call_args_list[0][0][0]
        self.assertIn("__Claim__", first_q)
        self.assertIn("CONSTRAINT", first_q.upper())
        # Second call persists rows
        second_kwargs = graph.query.call_args_list[1][1]
        self.assertEqual(second_kwargs["params"]["rows"], rows)

    def test_persist_claims_empty_is_noop(self):
        from src.graphrag.claims import persist_claims

        graph = MagicMock()
        self.assertEqual(persist_claims(graph, []), 0)
        graph.query.assert_not_called()

    def test_extract_claims_from_text_swallows_llm_errors(self):
        from src.graphrag.claims import extract_claims_from_text

        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("llm down")
        result = extract_claims_from_text(chain, "Some text about Acme", [{"id": "Acme"}])
        self.assertEqual(result, [])

    def test_extract_claims_entrypoint_with_mocks(self):
        from src.graphrag import claims as claims_mod

        graph = MagicMock()
        graph.query.side_effect = [
            # fetch chunks
            [
                {
                    "element_id": "e1",
                    "chunk_id": "c1",
                    "text": "Acme Corp acquired Widget Inc in 2020.",
                    "entities": [{"id": "Acme Corp", "description": "Company"}],
                }
            ],
            # CREATE CONSTRAINT
            None,
            # STORE_CLAIMS
            [{"stored": 1}],
        ]

        fake_chain = MagicMock()
        fake_chain.invoke.return_value = """
===CLAIM===
subject: Acme Corp
object: Widget Inc
type: acquisition
description: Acme acquired Widget
source_span: Acme Corp acquired Widget Inc in 2020
start_date: 2020-01-01
end_date:
status: known
"""

        result = claims_mod.extract_claims(
            graph, model="test-model", chain=fake_chain
        )
        self.assertEqual(result["chunks_processed"], 1)
        self.assertEqual(result["claims_extracted"], 1)
        self.assertEqual(result["claims_persisted"], 1)
        self.assertEqual(result["model"], "test-model")


class CommunityClaimsPrepareTests(unittest.TestCase):
    def test_prepare_string_includes_claims(self):
        from src.graphrag.helpers import prepare_community_string

        community = {
            "nodes": [
                {"id": "Acme", "type": "Org", "description": "company", "degree": 5},
                {"id": "Widget", "type": "Org", "description": "startup", "degree": 3},
            ],
            "rels": [
                {
                    "start": "Acme",
                    "type": "ACQUIRED",
                    "end": "Widget",
                    "description": "bought",
                    "combined_degree": 8,
                }
            ],
            "claims": [
                {
                    "id": "claim-1",
                    "subject": "Acme",
                    "object": "Widget",
                    "type": "acquisition",
                    "description": "Acme bought Widget in 2020",
                    "source_span": "Acme bought Widget",
                    "entity_ids": ["Acme", "Widget"],
                }
            ],
        }
        text = prepare_community_string(community, max_chars=5000)
        self.assertIn("Claims (covariates) are:", text)
        self.assertIn("acquisition", text)
        self.assertIn("Acme bought Widget in 2020", text)
        self.assertIn("ACQUIRED", text)

    def test_prepare_string_without_claims_unchanged_shape(self):
        from src.graphrag.helpers import prepare_community_string

        community = {
            "nodes": [{"id": "A", "type": "Person", "description": "x", "degree": 1}],
            "rels": [{"start": "A", "type": "KNOWS", "end": "B", "combined_degree": 2}],
        }
        text = prepare_community_string(community, max_chars=500)
        self.assertIn("Nodes are:", text)
        self.assertIn("Relationships are:", text)
        self.assertNotIn("Claims (covariates)", text)


if __name__ == "__main__":
    unittest.main()
