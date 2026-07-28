import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.reranker import OpenAIReranker, RerankConfig


def candidate(chunk_id, rrf_score, *, text=None, zone="standard_body"):
    return {
        "chunk_id": chunk_id,
        "rrf_score": rrf_score,
        "text": text or f"근거 {chunk_id}",
        "zone": zone,
        "standard_id": "1109",
    }


def ranking(chunk_id, relevance, body, direct, connected, reason="근거 있음"):
    return {
        "chunk_id": chunk_id,
        "question_relevance": relevance,
        "standard_body_priority": body,
        "direct_evidence": direct,
        "reference_connectivity": connected,
        "reason": reason,
    }


class FakeResponses:
    def __init__(self, payloads=None, error=None):
        self.payloads = list(payloads or [])
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=json.dumps(self.payloads.pop(0), ensure_ascii=False))


class RerankerTests(unittest.TestCase):
    def test_config_loads_rerank_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.yaml"
            path.write_text(
                "hybrid:\n  seed_top_k: 9\nrerank:\n  top_k: 3\n"
                "  weights:\n    direct_evidence: 0.5\n",
                encoding="utf-8",
            )
            config = RerankConfig.from_yaml(path)
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.direct_evidence_weight, 0.5)
        self.assertEqual(config.relevance_weight, 0.4)

    def test_structured_output_reranks_and_preserves_original_scores(self):
        responses = FakeResponses([{"rankings": [
            ranking("A", 2, 5, 2, 1, "키워드 일치"),
            ranking("B", 5, 4, 5, 3, "질문에 대한 직접 근거"),
        ]}])
        reranker = OpenAIReranker(
            SimpleNamespace(responses=responses), model="rerank-model",
            config=RerankConfig(top_k=2),
        )
        result = reranker.rerank("금융자산 제거 조건은?", [
            candidate("A", 0.04), candidate("B", 0.03),
        ])
        self.assertEqual([item["chunk_id"] for item in result], ["B", "A"])
        self.assertEqual(result[0]["rrf_score"], 0.03)
        self.assertEqual(result[0]["original_rank"], 2)
        self.assertEqual(result[0]["rerank_rank"], 1)
        self.assertEqual(result[0]["rerank_reason"], "질문에 대한 직접 근거")
        self.assertEqual(result[0]["rerank_status"], "scored")
        call = responses.calls[0]
        self.assertEqual(call["model"], "rerank-model")
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertFalse(call["store"])

    def test_batches_and_truncates_only_api_payload(self):
        responses = FakeResponses([
            {"rankings": [ranking("A", 4, 4, 4, 1)]},
            {"rankings": [ranking("B", 3, 3, 3, 1)]},
        ])
        original_text = "가" * 30
        reranker = OpenAIReranker(
            SimpleNamespace(responses=responses), model="model",
            config=RerankConfig(batch_size=1, max_candidate_chars=10, top_k=2),
        )
        result = reranker.rerank("질문", [
            candidate("A", 2, text=original_text), candidate("B", 1),
        ])
        self.assertEqual(len(responses.calls), 2)
        sent = json.loads(responses.calls[0]["input"][1]["content"])["candidates"][0]
        self.assertEqual(len(sent["text"]), 10)
        self.assertTrue(sent["text_truncated"])
        self.assertEqual(next(item for item in result if item["chunk_id"] == "A")["text"], original_text)

    def test_unknown_missing_or_duplicate_output_falls_back_to_input_order(self):
        invalid_payloads = [
            {"rankings": [ranking("A", 5, 5, 5, 5)]},
            {"rankings": [ranking("A", 5, 5, 5, 5), ranking("A", 4, 4, 4, 4)]},
            {"rankings": [ranking("A", 5, 5, 5, 5), ranking("X", 4, 4, 4, 4)]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                responses = FakeResponses([payload])
                reranker = OpenAIReranker(SimpleNamespace(responses=responses), model="model")
                result = reranker.rerank("질문", [candidate("A", 2), candidate("B", 1)])
                self.assertEqual([item["chunk_id"] for item in result], ["A", "B"])
                self.assertTrue(all(item["rerank_status"] == "fallback" for item in result))
                self.assertTrue(all(item["rerank_score"] is None for item in result))

    def test_api_failure_falls_back_and_deduplicates_input(self):
        responses = FakeResponses(error=RuntimeError("temporary failure"))
        reranker = OpenAIReranker(SimpleNamespace(responses=responses), model="model")
        result = reranker.rerank(
            "질문", [candidate("A", 2), candidate("A", 1), candidate("B", 0)], top_k=2
        )
        self.assertEqual([item["chunk_id"] for item in result], ["A", "B"])
        self.assertEqual([item["original_rank"] for item in result], [1, 2])
        self.assertTrue(all("RuntimeError" in item["rerank_reason"] for item in result))

    def test_score_tie_break_is_deterministic(self):
        responses = FakeResponses([{"rankings": [
            ranking("A", 4, 4, 3, 2),
            ranking("B", 3, 4, 4, 3),
            ranking("C", 4, 4, 3, 2),
        ]}])
        reranker = OpenAIReranker(SimpleNamespace(responses=responses), model="model")
        result = reranker.rerank(
            "질문", [candidate("A", 3), candidate("B", 2), candidate("C", 1)]
        )
        self.assertEqual([item["chunk_id"] for item in result], ["B", "A", "C"])

    def test_empty_question_and_missing_id_are_rejected(self):
        reranker = OpenAIReranker(SimpleNamespace(), model="model")
        with self.assertRaises(ValueError):
            reranker.rerank("  ", [candidate("A", 1)])
        with self.assertRaises(ValueError):
            reranker.rerank("질문", [{"text": "근거"}])


if __name__ == "__main__":
    unittest.main()
