import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.qa_pipeline import AccountingQAPipeline  # noqa: E402


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {"results": self.results, "seed_count": len(self.results)}


class StubGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, question, candidates):
        self.calls.append((question, candidates))
        return self.answer


def candidate(chunk_id="C1", citation="문단 1", **metadata):
    return {"chunk_id": chunk_id, "text": "근거", "citation_label": citation, **metadata}


def answer(conclusion="결론이다. [E1]", evidence_ids=("E1",)):
    return {
        "conclusion": conclusion,
        "reasoning": ["판단 단계 [E1]"],
        "evidence": [{"evidence_id": value, "citation": "문단 1", "statement": "근거"}
                     for value in evidence_ids],
    }


class QAPipelineTests(unittest.TestCase):
    def test_grounded_answer_is_returned_as_answered(self):
        retriever = StubRetriever([candidate()])
        generator = StubGenerator(answer())
        result = AccountingQAPipeline(retriever, generator).ask("금융자산의 정의는?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["reason"], "sufficient")
        self.assertEqual(result["answer"]["conclusion"], "결론이다. [E1]")
        self.assertEqual(len(generator.calls), 1)

    def test_generator_declining_is_reported_as_insufficient(self):
        generator = StubGenerator(answer(conclusion="근거 부족: 직접 규정한 문단이 없습니다.",
                                         evidence_ids=()))
        result = AccountingQAPipeline(StubRetriever([candidate()]), generator).ask("질문")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "self_declined")

    def test_alternate_decline_prefix_is_recognised(self):
        generator = StubGenerator(answer(conclusion="답변 불가: 관련 문단이 없습니다.",
                                         evidence_ids=()))
        result = AccountingQAPipeline(StubRetriever([candidate()]), generator).ask("질문")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "self_declined")

    def test_empty_retrieval_skips_generation(self):
        generator = StubGenerator(answer())
        result = AccountingQAPipeline(StubRetriever([]), generator).ask("질문")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "no_evidence_found")
        self.assertEqual(result["answer"]["evidence"], [])
        self.assertEqual(generator.calls, [])

    def test_filters_are_passed_through_to_the_retriever(self):
        retriever = StubRetriever([candidate()])
        AccountingQAPipeline(retriever, StubGenerator(answer())).ask(
            "  질문  ", standard_id="1109", zone="standard_body", top_k=5,
        )
        question, kwargs = retriever.calls[0]
        self.assertEqual(question, "질문")
        self.assertEqual(kwargs["standard_id"], "1109")
        self.assertEqual(kwargs["zone"], "standard_body")
        self.assertEqual(kwargs["top_k"], 5)

    def test_answer_evidence_is_enriched_with_retrieval_metadata(self):
        retriever = StubRetriever([
            candidate(pdf_page_start=12, candidate_source="hybrid"),
        ])
        result = AccountingQAPipeline(retriever, StubGenerator(answer())).ask("질문")
        evidence = result["answer"]["evidence"][0]
        self.assertEqual(evidence["evidence_id"], "E1")
        self.assertEqual(evidence["pdf_page_start"], 12)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            AccountingQAPipeline(StubRetriever([]), StubGenerator(answer())).ask("   ")


if __name__ == "__main__":
    unittest.main()
