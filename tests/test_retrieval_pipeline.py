import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.pipeline import (  # noqa: E402
    RetrievalPipeline,
    build_rerank_candidates,
)


class FakeHybridRetriever:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, question, **filters):
        self.calls.append((question, filters))
        return self.rows


class FakeGraphExpander:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def expand(self, question, seeds):
        self.calls.append((question, seeds))
        return self.rows


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, question, candidates, *, top_k=None):
        self.calls.append((question, candidates, top_k))
        return [{**candidate, "rerank_rank": rank}
                for rank, candidate in enumerate(candidates[:top_k], 1)]


def graph_row(node_id, *, hop=1, text="graph text", standard_id="1109", zone=None):
    return {
        "node_id": node_id,
        "node_labels": ["Paragraph"],
        "text": text,
        "citation": "5.5.17",
        "standard_id": standard_id,
        "zone": zone,
        "hop": hop,
        "graph_score": 0.25,
        "seed_chunk_ids": ["C1"],
        "paths": [[{"edge_type": "REFERS_TO", "from_id": "A", "to_id": node_id}]],
    }


class RetrievalPipelineTests(unittest.TestCase):
    def test_runs_stages_in_order_and_preserves_graph_provenance(self):
        seeds = [{"chunk_id": "C1", "text": "seed", "rrf_score": 0.03,
                  "standard_id": "1109", "zone": "standard_body"}]
        evidence = [graph_row("KIFRS1109-5.5.17", zone="standard_body")]
        hybrid = FakeHybridRetriever(seeds)
        graph = FakeGraphExpander(evidence)
        reranker = FakeReranker()

        output = RetrievalPipeline(hybrid, graph, reranker).retrieve(
            "  expected loss?  ", standard_id="KIFRS1109",
            zone="standard_body", top_k=2,
        )

        self.assertEqual(hybrid.calls, [
            ("expected loss?", {"standard_id": "1109", "zone": "standard_body"})
        ])
        self.assertEqual(graph.calls, [("expected loss?", seeds)])
        question, candidates, top_k = reranker.calls[0]
        self.assertEqual(question, "expected loss?")
        self.assertEqual(top_k, 2)
        self.assertEqual([item["chunk_id"] for item in candidates], [
            "C1", "GRAPH::KIFRS1109-5.5.17"
        ])
        graph_candidate = candidates[1]
        self.assertEqual(graph_candidate["graph_node_id"], "KIFRS1109-5.5.17")
        self.assertEqual(graph_candidate["graph_hop"], 1)
        self.assertEqual(graph_candidate["graph_score"], 0.25)
        self.assertEqual(graph_candidate["graph_path"][0]["edge_type"], "REFERS_TO")
        self.assertEqual(output["filters"], {
            "standard_id": "1109", "zone": "standard_body"
        })
        self.assertEqual(output["result_count"], 2)

    def test_excludes_hop_zero_empty_text_and_duplicate_graph_nodes(self):
        rows = [
            graph_row("SOURCE", hop=0),
            graph_row("EMPTY", text="  "),
            graph_row("TARGET"),
            graph_row("TARGET", text="duplicate"),
        ]
        candidates = build_rerank_candidates(
            [{"chunk_id": "C1"}, {"chunk_id": "C1"}], rows
        )
        self.assertEqual([item["chunk_id"] for item in candidates], [
            "C1", "GRAPH::TARGET"
        ])

    def test_graph_candidates_obey_standard_and_zone_filters(self):
        rows = [
            graph_row("KEEP", standard_id="1109", zone="standard_body"),
            graph_row("WRONG-STANDARD", standard_id="1032", zone="standard_body"),
            graph_row("WRONG-ZONE", standard_id="1109", zone="appendix"),
            graph_row("UNKNOWN-ZONE", standard_id="1109", zone=None),
        ]
        candidates = build_rerank_candidates(
            [], rows, standard_id="1109", zone="standard_body"
        )
        self.assertEqual([item["chunk_id"] for item in candidates], ["GRAPH::KEEP"])

    def test_empty_question_stops_before_dependencies(self):
        hybrid = FakeHybridRetriever([])
        graph = FakeGraphExpander([])
        reranker = FakeReranker()
        with self.assertRaises(ValueError):
            RetrievalPipeline(hybrid, graph, reranker).retrieve("  ")
        self.assertEqual(hybrid.calls, [])
        self.assertEqual(graph.calls, [])
        self.assertEqual(reranker.calls, [])


if __name__ == "__main__":
    unittest.main()
