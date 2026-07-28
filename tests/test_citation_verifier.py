import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.generation.citation_verifier import verify_citations  # noqa: E402


def answer(*evidence, status="sufficient", conclusion="결론", reasoning="판단 과정"):
    return {
        "status": status,
        "conclusion": conclusion,
        "reasoning": reasoning,
        "evidence": list(evidence),
    }


class CitationVerifierTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"chunk_id": "C1", "citation_label": "제1109호 문단 5.5.17"},
            {"chunk_id": "GRAPH::P2", "citation": "제1109호 문단 B5.5.51"},
            {"evidence_id": "E3", "citation_label": "제1032호 문단 16"},
        ]

    def test_validates_hybrid_and_graph_citations_and_reports_unused_ids(self):
        result = verify_citations(
            answer(
                {"evidence_id": "C1", "citation": "제1109호 문단 5.5.17", "statement": "측정한다."},
                {"evidence_id": "GRAPH::P2", "citation": "제1109호 문단 B5.5.51", "statement": "정보를 고려한다."},
            ),
            {"results": self.candidates},
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, ())
        self.assertEqual(result.used_evidence_ids, ("C1", "GRAPH::P2"))
        self.assertEqual(result.unused_evidence_ids, ("E3",))
        self.assertEqual(result.to_dict()["unused_evidence_ids"], ["E3"])

    def test_rejects_unknown_id_citation_mismatch_and_empty_statement(self):
        result = verify_citations(
            answer(
                {"evidence_id": "C1", "citation": "틀린 인용", "statement": ""},
                {"evidence_id": "MISSING", "citation": "아무 문단", "statement": "문장"},
            ),
            self.candidates,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("citation mismatch for C1" in error for error in result.errors))
        self.assertIn("evidence statement is empty: C1", result.errors)
        self.assertIn("unknown answer evidence_id: MISSING", result.errors)

    def test_rejects_evidence_when_retrieval_candidate_has_no_citation(self):
        result = verify_citations(
            answer({"evidence_id": "C1", "citation": "", "statement": "근거"}),
            [{"chunk_id": "C1", "text": "원문"}],
        )
        self.assertFalse(result.valid)
        self.assertIn("retrieval evidence has no citation: C1", result.errors)

    def test_rejects_duplicate_answer_and_candidate_ids_deterministically(self):
        result = verify_citations(
            answer(
                {"evidence_id": "C1", "citation": "제1109호 문단 5.5.17", "statement": "첫째"},
                {"evidence_id": "C1", "citation": "제1109호 문단 5.5.17", "statement": "둘째"},
            ),
            [self.candidates[0], dict(self.candidates[0])],
        )
        self.assertFalse(result.valid)
        self.assertIn("duplicate retrieval evidence_id: C1", result.errors)
        self.assertIn("duplicate answer evidence_id: C1", result.errors)
        self.assertEqual(result.used_evidence_ids, ("C1",))

    def test_requires_conclusion_and_reasoning(self):
        result = verify_citations(
            answer(status="sufficient", conclusion=" ", reasoning=None),
            self.candidates,
        )
        self.assertFalse(result.valid)
        self.assertIn("answer conclusion is empty", result.errors)
        self.assertIn("answer reasoning is empty", result.errors)
        self.assertIn("sufficient answer has no evidence", result.errors)

    def test_insufficient_answer_may_have_no_evidence_but_needs_explanation(self):
        result = verify_citations(
            answer(status="insufficient", conclusion="판단 불가", reasoning="검색 근거 부족"),
            self.candidates,
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.used_evidence_ids, ())
        self.assertEqual(result.unused_evidence_ids, ("C1", "GRAPH::P2", "E3"))

    def test_three_key_generator_answer_and_prepared_catalog_are_supported(self):
        generated = {
            "conclusion": "기대신용손실을 측정한다. [E1]",
            "reasoning": ["관련 문단이 측정 원칙을 규정한다. [E1]"],
            "evidence": [{
                "evidence_id": "E1",
                "citation": "제1109호 문단 5.5.17",
                "statement": "측정 원칙",
            }],
        }
        catalog = [{
            "evidence_id": "E1",
            "source_id": "C1",
            "citation": "제1109호 문단 5.5.17",
            "statement": "원문",
        }]
        result = verify_citations(generated, catalog)
        self.assertTrue(result.valid)
        self.assertEqual(result.used_evidence_ids, ("E1",))

    def test_three_key_insufficient_answer_is_inferred_from_exact_prefix(self):
        generated = {
            "conclusion": "근거 부족: 필요한 조건을 확인할 수 없습니다.",
            "reasoning": ["근거 부족: 관련 문단이 검색되지 않았습니다."],
            "evidence": [],
        }
        self.assertTrue(verify_citations(generated, self.candidates).valid)

    def test_empty_reasoning_list_is_rejected(self):
        generated = {
            "conclusion": "결론",
            "reasoning": [],
            "evidence": [{
                "evidence_id": "C1",
                "citation": "제1109호 문단 5.5.17",
                "statement": "근거",
            }],
        }
        result = verify_citations(generated, self.candidates)
        self.assertFalse(result.valid)
        self.assertIn("answer reasoning is empty", result.errors)

    def test_insufficient_answer_still_validates_any_supplied_evidence(self):
        result = verify_citations(
            answer(
                {"evidence_id": "E3", "citation": "오류", "statement": ""},
                status="불충분",
            ),
            self.candidates,
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("citation mismatch for E3" in error for error in result.errors))
        self.assertIn("evidence statement is empty: E3", result.errors)

    def test_rejects_unsupported_status_and_non_sequence_evidence(self):
        result = verify_citations(
            {"status": "maybe", "conclusion": "결론", "reasoning": "이유", "evidence": {}},
            self.candidates,
        )
        self.assertFalse(result.valid)
        self.assertIn("unsupported answer status: maybe", result.errors)
        self.assertIn("answer evidence must be a sequence", result.errors)


if __name__ == "__main__":
    unittest.main()
