import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.graph_expansion import (
    GraphExpander, GraphExpansionConfig, question_terms,
)


class FakeSession:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return self.results.pop(0)


class FakeDriver:
    def __init__(self, session):
        self.fake_session = session
        self.database = None
    def session(self, *, database):
        self.database = database
        return self.fake_session


def source_row(label="Paragraph", node_id="KIFRS1109-5.5.17"):
    return {
        "seed_chunk_id": "C1", "seed_rank": 1, "seed_score": 0.02,
        "source_element_id": "e-source", "source_id": node_id,
        "source_labels": [label], "source_text": "기대신용손실 측정",
        "source_number": "5.5.17", "source_standard_id": "1109",
        "source_zone": "standard_body",
        "edge_properties": {"provenance": "chunk_builder", "confidence": 1.0,
                            "review_status": "approved"},
    }


def neighbor_row(edge="REFERS_TO", target="KIFRS1109-5.5.18",
                 target_label="Paragraph", text="미래전망정보를 고려한다",
                 source="KIFRS1109-5.5.17", element="e-target"):
    return {
        "frontier_key": "source:0", "source_element_id": "e-source",
        "source_id": source, "source_labels": ["Paragraph"],
        "target_element_id": element, "target_id": target,
        "target_labels": [target_label], "target_text": text,
        "target_title": None, "target_number": "5.5.18",
        "target_standard_id": "1109", "target_zone": "standard_body",
        "edge_type": edge,
        "direction": "outgoing", "edge_properties": {
            "provenance": "explicit_reference_parser", "confidence": 1.0,
            "review_status": "approved"},
    }


class GraphExpansionTests(unittest.TestCase):
    def test_config_loads_graph_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.yaml"
            path.write_text(
                "hybrid:\n  seed_top_k: 3\ngraph:\n  max_nodes: 9\n"
                "  edge_allowlist: [REFERS_TO, NEXT]\n"
                "  second_hop:\n    lexical_overlap_min: 2\n", encoding="utf-8",
            )
            config = GraphExpansionConfig.from_yaml(path)
        self.assertEqual(config.max_nodes, 9)
        self.assertEqual(config.edge_allowlist, ("REFERS_TO", "NEXT"))
        self.assertEqual(config.lexical_overlap_min, 2)

    def test_config_rejects_unbounded_or_unknown_traversal(self):
        with self.assertRaises(ValueError):
            GraphExpansionConfig(max_hops=3)
        with self.assertRaises(ValueError):
            GraphExpansionConfig(edge_allowlist=("APPEARS_ON",))

    def test_question_terms_are_stable_and_deduplicated(self):
        self.assertEqual(question_terms("기대신용손실 측정, 기대신용손실!"),
                         ("기대신용손실", "측정"))

    def test_reference_opens_second_hop_and_preserves_full_path(self):
        first = neighbor_row()
        second = neighbor_row(edge="CONTAINS", target="KIFRS1109-5.5.18-S01",
                              target_label="Subparagraph", text="합리적인 정보",
                              source="KIFRS1109-5.5.18", element="e-detail")
        second["frontier_key"] = "first:0"
        second["source_element_id"] = "e-target"
        session = FakeSession([[source_row()], [first], [second]])
        expander = GraphExpander(FakeDriver(session), database="neo4j")
        result = expander.expand("기대신용손실은 어떻게 측정하는가?",
                                 [{"chunk_id": "C1", "rrf_score": 0.02}])
        detail = next(item for item in result if item["node_id"].endswith("S01"))
        self.assertEqual(detail["hop"], 2)
        self.assertEqual(detail["zone"], "standard_body")
        self.assertEqual([edge["edge_type"] for edge in detail["paths"][0]],
                         ["DERIVED_FROM", "REFERS_TO", "CONTAINS"])
        self.assertEqual(detail["seed_chunk_ids"], ["C1"])
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(session.calls[2][1]["frontier"][0]["element_id"], "e-target")

    def test_plain_next_without_overlap_stays_at_one_hop(self):
        first = neighbor_row(edge="NEXT", text="전혀 다른 주제")
        session = FakeSession([[source_row()], [first]])
        expander = GraphExpander(FakeDriver(session), database="neo4j")
        result = expander.expand("기대신용손실 측정",
                                 [{"chunk_id": "C1", "rrf_score": 0.02}])
        self.assertEqual(max(item["hop"] for item in result), 1)
        self.assertEqual(len(session.calls), 2)

    def test_detail_to_parent_can_restore_hierarchy_at_second_hop(self):
        source = source_row(label="Subparagraph", node_id="KIFRS1109-5.5.17-S01")
        first = neighbor_row(edge="CONTAINS", target="KIFRS1109-5.5.17",
                             target_label="Paragraph", text="다른 문구",
                             source="KIFRS1109-5.5.17-S01")
        first["direction"] = "incoming"
        session = FakeSession([[source], [first], []])
        expander = GraphExpander(FakeDriver(session), database="neo4j")
        expander.expand("신용위험", [{"chunk_id": "C1", "rrf_score": 0.02}])
        self.assertEqual(len(session.calls), 3)

    def test_semantic_mention_opens_second_hop_to_definition(self):
        first = neighbor_row(edge="MENTIONS", target="CONCEPT-ECL",
                             target_label="Concept", text="기대신용손실의 정의",
                             source="KIFRS1109-5.5.17", element="e-concept")
        first["target_title"] = "기대신용손실"
        second = neighbor_row(edge="MENTIONS", target="KIFRS1109-T-0014",
                              target_label="Table", text="기대신용손실 정의표",
                              source="CONCEPT-ECL", element="e-definition")
        second["frontier_key"] = "first:0"
        second["source_element_id"] = "e-concept"
        second["direction"] = "incoming"
        second["edge_properties"]["role"] = "definition"
        session = FakeSession([[source_row()], [first], [second]])
        result = GraphExpander(FakeDriver(session), database="neo4j").expand(
            "기대신용손실이란 무엇인가?", [{"chunk_id": "C1", "rrf_score": 0.02}]
        )
        definition = next(item for item in result if item["node_id"] == "KIFRS1109-T-0014")
        self.assertEqual([edge["edge_type"] for edge in definition["paths"][0]],
                         ["DERIVED_FROM", "MENTIONS", "MENTIONS"])
        self.assertFalse(any("Concept" in item["node_labels"] for item in result))
        self.assertIn("edge.role = 'definition'", session.calls[1][0])

    def test_concept_is_not_returned_when_reached_on_second_hop(self):
        first = neighbor_row(edge="HAS_TABLE", target="KIFRS1109-T-0014",
                             target_label="Table", text="공식 정의표",
                             source="KIFRS1109-5.5.17", element="e-table")
        second = neighbor_row(edge="MENTIONS", target="CONCEPT-ECL",
                              target_label="Concept", text="기대신용손실 정의",
                              source="KIFRS1109-T-0014", element="e-concept")
        second["frontier_key"] = "first:0"
        second["source_element_id"] = "e-table"
        session = FakeSession([[source_row()], [first], [second]])
        result = GraphExpander(FakeDriver(session), database="neo4j").expand(
            "기대신용손실 정의", [{"chunk_id": "C1", "rrf_score": 0.02}]
        )
        self.assertFalse(any("Concept" in item["node_labels"] for item in result))

    def test_queries_enforce_allowlist_approval_and_outgoing_references(self):
        session = FakeSession([[source_row()], []])
        driver = FakeDriver(session)
        config = GraphExpansionConfig(edge_allowlist=("REFERS_TO", "NEXT"), max_hops=1)
        GraphExpander(driver, database="neo4j", config=config).expand(
            "금융자산 제거", [{"chunk_id": "C1", "rrf_score": 0.02}]
        )
        self.assertEqual(driver.database, "neo4j")
        source_query, source_parameters = session.calls[0]
        neighbor_query, neighbor_parameters = session.calls[1]
        self.assertIn("DERIVED_FROM", source_query)
        self.assertIn("edge.review_status = 'approved'", source_query)
        self.assertIn("type(edge) IN $edge_allowlist", neighbor_query)
        self.assertIn("startNode(edge) = source", neighbor_query)
        self.assertEqual(neighbor_parameters["edge_allowlist"], ["REFERS_TO", "NEXT"])

    def test_empty_seeds_avoid_database_call(self):
        session = FakeSession([])
        result = GraphExpander(FakeDriver(session), database="neo4j").expand("질문", [])
        self.assertEqual(result, [])
        self.assertEqual(session.calls, [])

    def test_global_node_limit_is_deterministic(self):
        first = [
            neighbor_row(edge="NEXT", target="B", text="질문 B", element="e-b"),
            neighbor_row(edge="NEXT", target="A", text="질문 A", element="e-a"),
        ]
        session = FakeSession([[source_row()], first])
        config = GraphExpansionConfig(max_nodes=2, max_hops=1)
        result = GraphExpander(FakeDriver(session), database="neo4j", config=config).expand(
            "질문", [{"chunk_id": "C1", "rrf_score": 0.02}]
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["node_id"], "KIFRS1109-5.5.17")


if __name__ == "__main__":
    unittest.main()
