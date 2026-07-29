import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.hybrid import (
    HybridConfig, HybridRetriever, escape_lucene_query, normalize_standard_id,
    reciprocal_rank_fusion,
)


def row(chunk_id, score):
    return {"chunk_id": chunk_id, "score": score, "standard_id": "1109",
            "zone": "standard_body", "text": chunk_id,
            "contextualized_text": chunk_id, "citation_label": f"문단 {chunk_id}",
            "chunk_type": "paragraph", "pdf_page_start": 1,
            "pdf_page_end": 1, "search_priority": 3}


class FakeSession:
    def __init__(self, dense, sparse):
        self.results, self.calls = [dense, sparse], []
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return self.results.pop(0)


class FakeDriver:
    def __init__(self, session): self.fake_session, self.database = session, None
    def session(self, *, database):
        self.database = database
        return self.fake_session


class FakeEmbeddings:
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])


class HybridTests(unittest.TestCase):
    def test_lucene_escape(self):
        self.assertEqual(
            escape_lucene_query('상각후원가+(제거) && "위험"/보상?'),
            '상각후원가\\+\\(제거\\) \\&& \\"위험\\"\\/보상\\?',
        )

    def test_standard_id_normalization(self):
        self.assertEqual(normalize_standard_id("1109"), "1109")
        self.assertEqual(normalize_standard_id("KIFRS1109"), "1109")
        self.assertEqual(normalize_standard_id("K-IFRS 1109"), "1109")
        self.assertEqual(normalize_standard_id("제1109호"), "1109")
        self.assertIsNone(normalize_standard_id("  "))
        with self.assertRaises(ValueError):
            normalize_standard_id("IFRS9")

    def test_rrf_preserves_scores_and_deterministic_tie_break(self):
        result = reciprocal_rank_fusion(
            [row("B", 0.9), row("A", 0.8)],
            [row("A", 12.0), row("B", 11.0)], rrf_k=60, top_k=10,
        )
        self.assertEqual([item["chunk_id"] for item in result], ["A", "B"])
        self.assertEqual((result[0]["dense_rank"], result[0]["dense_score"]), (2, 0.8))
        self.assertEqual((result[0]["sparse_rank"], result[0]["sparse_score"]), (1, 12.0))
        self.assertEqual(result[0]["source_channels"], ["dense", "sparse"])
        self.assertAlmostEqual(result[0]["rrf_score"], 1 / 62 + 1 / 61)

    def test_weighted_rrf_and_limit(self):
        result = reciprocal_rank_fusion(
            [row("D", 0.9)], [row("S", 5.0)], rrf_k=60,
            dense_weight=2.0, sparse_weight=1.0, top_k=1,
        )
        self.assertEqual(result[0]["chunk_id"], "D")
        self.assertAlmostEqual(result[0]["rrf_score"], 2 / 61)

    def test_config_loads_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.yaml"
            path.write_text("hybrid:\n  dense_top_k: 7\n  sparse_weight: 1.5\n", encoding="utf-8")
            config = HybridConfig.from_yaml(path)
        self.assertEqual(config.dense_top_k, 7)
        self.assertEqual(config.sparse_top_k, 50)
        self.assertEqual(config.sparse_weight, 1.5)

    def test_config_loads_optional_score_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.yaml"
            path.write_text(
                "hybrid:\n  dense_min_score: 0.75\n  sparse_min_score: 3.5\n",
                encoding="utf-8",
            )
            config = HybridConfig.from_yaml(path)
        self.assertEqual(config.dense_min_score, 0.75)
        self.assertEqual(config.sparse_min_score, 3.5)

    def test_search_uses_injected_clients_and_parameters(self):
        session = FakeSession([row("D", 0.9)], [row("S", 4.0)])
        embeddings = FakeEmbeddings()
        driver = FakeDriver(session)
        retriever = HybridRetriever(
            driver, SimpleNamespace(embeddings=embeddings), database="neo4j",
            embedding_model="text-embedding-3-large",
        )
        result = retriever.search("금융자산+(제거)?", standard_id="1109", zone="standard_body")
        self.assertEqual(driver.database, "neo4j")
        self.assertEqual(embeddings.calls, [{"model": "text-embedding-3-large",
            "input": ["금융자산+(제거)?"], "dimensions": 3072,
            "encoding_format": "float"}])
        dense_query, dense_parameters = session.calls[0]
        sparse_query, sparse_parameters = session.calls[1]
        self.assertIn("VECTOR INDEX chunk_embedding_vector", dense_query)
        self.assertIn("coalesce(node.searchable, false) = true", dense_query)
        self.assertIn("db.index.fulltext.queryNodes('chunk_fulltext'", sparse_query)
        self.assertIn("coalesce(node.inactive, false) = false", sparse_query)
        self.assertEqual(dense_parameters["embedding"], [0.1, 0.2])
        self.assertEqual(dense_parameters["standard_id"], "1109")
        self.assertEqual(sparse_parameters["zone"], "standard_body")
        self.assertEqual(sparse_parameters["lucene_query"], "금융자산\\+\\(제거\\)\\?")
        self.assertEqual({item["chunk_id"] for item in result}, {"D", "S"})

    def test_search_accepts_a_separate_sparse_query(self):
        session = FakeSession([row("D", 0.9)], [row("S", 4.0)])
        embeddings = FakeEmbeddings()
        retriever = HybridRetriever(
            FakeDriver(session), SimpleNamespace(embeddings=embeddings),
            database="neo4j", embedding_model="model",
        )

        retriever.search(
            "금융부채 조건변경의 회계처리",
            sparse_query="금융부채 조건변경 유효이자율",
        )

        self.assertEqual(
            embeddings.calls[0]["input"], ["금융부채 조건변경의 회계처리"],
        )
        self.assertEqual(
            session.calls[1][1]["lucene_query"],
            "금융부채 조건변경 유효이자율",
        )

    def test_thresholds_filter_each_channel_before_rrf(self):
        session = FakeSession(
            [row("D-low", 0.7), row("D-pass", 0.9)],
            [row("S-pass", 5.0), row("S-low", 2.0)],
        )
        retriever = HybridRetriever(
            FakeDriver(session), SimpleNamespace(embeddings=FakeEmbeddings()),
            database="neo4j", embedding_model="model",
            config=HybridConfig(dense_min_score=0.8, sparse_min_score=4.0),
        )
        result = retriever.search("질문")
        self.assertEqual({item["chunk_id"] for item in result}, {"D-pass", "S-pass"})

    def test_opt_in_score_log_contains_all_raw_rows_without_question(self):
        session = FakeSession([row("D", 0.9)], [row("S", 4.0)])
        retriever = HybridRetriever(
            FakeDriver(session), SimpleNamespace(embeddings=FakeEmbeddings()),
            database="neo4j", embedding_model="model",
        )
        environment = {
            "RETRIEVAL_SCORE_LOG": "1",
            "RETRIEVAL_SCORE_LOG_INCLUDE_QUESTION": "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            with self.assertLogs("accounting_rag.retrieval.scores", level="INFO") as captured:
                retriever.search("금융자산")
        payload = json.loads(captured.records[0].getMessage())
        self.assertEqual(payload["event"], "retrieval_scores")
        self.assertEqual(payload["dense"][0]["score"], 0.9)
        self.assertEqual(payload["sparse"][0]["score"], 4.0)
        self.assertNotIn("question", payload)

    def test_search_with_scores_exposes_raw_channels_for_calibration(self):
        session = FakeSession([row("D", 0.9)], [row("S", 4.0)])
        retriever = HybridRetriever(
            FakeDriver(session), SimpleNamespace(embeddings=FakeEmbeddings()),
            database="neo4j", embedding_model="model",
        )
        snapshot = retriever.search_with_scores("금융자산")
        self.assertEqual(snapshot["dense"][0]["chunk_id"], "D")
        self.assertEqual(snapshot["sparse"][0]["chunk_id"], "S")
        self.assertEqual({row["chunk_id"] for row in snapshot["results"]}, {"D", "S"})

    def test_empty_question_fails_before_external_calls(self):
        session = FakeSession([], [])
        retriever = HybridRetriever(
            FakeDriver(session), SimpleNamespace(embeddings=FakeEmbeddings()),
            database="neo4j", embedding_model="model",
        )
        with self.assertRaises(ValueError): retriever.search("   ")
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()
