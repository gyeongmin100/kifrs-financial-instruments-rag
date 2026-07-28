import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from accounting_rag.generation.sufficiency import (
    EvidenceSufficiencyChecker,
    OpenAISemanticJudge,
    SufficiencyConfig,
)


def evidence(evidence_id, *, relevance=4, direct=4, text="근거", citation="1109.1"):
    return {
        "chunk_id": evidence_id,
        "text": text,
        "citation_label": citation,
        "rerank_status": "scored",
        "rerank_scores": {
            "question_relevance": relevance,
            "direct_evidence": direct,
        },
    }


def semantic_payload(*ids, sufficient=True, confidence=0.9, missing=None):
    return {
        "claim_coverage": [{
            "claim": "핵심 판단",
            "covered": sufficient,
            "evidence_ids": list(ids),
        }],
        "supported_evidence_ids": list(ids),
        "missing_aspects": list(missing or []),
        "sufficient": sufficient,
        "confidence": confidence,
    }


class FakeJudge:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def judge(self, question, candidates):
        self.calls.append((question, candidates))
        if self.error:
            raise self.error
        return self.payload


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.payload, ensure_ascii=False))


class SufficiencyTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [evidence("A"), evidence("B"), evidence("C")]

    def test_config_loads_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answering.yaml"
            path.write_text(
                "sufficiency:\n  min_candidates: 5\n"
                "  min_direct_evidence: 4\n  semantic_confidence_threshold: 0.8\n",
                encoding="utf-8",
            )
            config = SufficiencyConfig.from_yaml(path)
        self.assertEqual(config.min_candidates, 5)
        self.assertEqual(config.min_direct_evidence, 4)
        self.assertEqual(config.semantic_confidence_threshold, 0.8)

    def test_deterministic_failure_skips_semantic_judge(self):
        judge = FakeJudge(semantic_payload("A", "B"))
        result = EvidenceSufficiencyChecker(judge).check("질문", [
            evidence("A", text="", citation=""),
            evidence("B", relevance=1, direct=1),
        ])
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["status"], "deterministic_failed")
        self.assertEqual(judge.calls, [])
        self.assertFalse(result["deterministic"]["checks"]["candidate_count"])

    def test_graph_node_id_is_a_valid_citation(self):
        graph = evidence("GRAPH::P1", citation="")
        graph["graph_node_id"] = "P1"
        judge = FakeJudge(semantic_payload("A", "GRAPH::P1"))
        result = EvidenceSufficiencyChecker(judge).check(
            "질문", [evidence("A"), graph, evidence("C")]
        )
        self.assertTrue(result["sufficient"])

    def test_only_qualified_evidence_is_sent_to_judge(self):
        judge = FakeJudge(semantic_payload("A", "B"))
        candidates = [evidence("A"), evidence("B"), evidence("C", direct=1)]
        result = EvidenceSufficiencyChecker(judge).check("질문", candidates)
        self.assertTrue(result["sufficient"])
        self.assertEqual([item["chunk_id"] for item in judge.calls[0][1]], ["A", "B"])

    def test_missing_aspect_or_low_confidence_is_insufficient(self):
        cases = [
            semantic_payload("A", "B", missing=["예외 조건"]),
            semantic_payload("A", "B", confidence=0.4),
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                result = EvidenceSufficiencyChecker(FakeJudge(payload)).check(
                    "질문", self.candidates
                )
                self.assertFalse(result["sufficient"])
                self.assertEqual(result["status"], "semantic_insufficient")

    def test_hallucinated_id_is_rejected_conservatively(self):
        result = EvidenceSufficiencyChecker(
            FakeJudge(semantic_payload("A", "INVENTED"))
        ).check("질문", self.candidates)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["status"], "semantic_error")
        self.assertEqual(result["semantic"]["error_type"], "ValueError")

    def test_api_failure_is_conservative(self):
        result = EvidenceSufficiencyChecker(
            FakeJudge(error=RuntimeError("temporary"))
        ).check("질문", self.candidates)
        self.assertFalse(result["sufficient"])
        self.assertEqual(result["status"], "semantic_error")
        self.assertEqual(result["confidence"], 0.0)

    def test_openai_judge_uses_strict_schema_without_storage(self):
        responses = FakeResponses(semantic_payload("A", "B"))
        judge = OpenAISemanticJudge(
            SimpleNamespace(responses=responses), model="judge-model"
        )
        result = judge.judge("질문", [evidence("A"), evidence("B")])
        self.assertTrue(result["sufficient"])
        call = responses.calls[0]
        self.assertEqual(call["model"], "judge-model")
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertFalse(call["store"])

    def test_empty_question_is_rejected_and_missing_judge_is_conservative(self):
        checker = EvidenceSufficiencyChecker(None)
        with self.assertRaises(ValueError):
            checker.check("  ", self.candidates)
        result = checker.check("질문", self.candidates)
        self.assertEqual(result["status"], "semantic_not_run")
        self.assertFalse(result["sufficient"])


if __name__ == "__main__":
    unittest.main()
