import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.qa_pipeline import AccountingQAPipeline  # noqa: E402


class StubRetriever:
    def __init__(self, results, **metadata):
        self.results = results
        self.metadata = metadata
        self.calls = []

    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {"results": self.results, "seed_count": len(self.results), **self.metadata}


class StubGenerator:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, question, candidates, **kwargs):
        self.calls.append((question, candidates, kwargs))
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
        self.assertEqual(result["reason"], "evidence_available")
        self.assertEqual(result["answer"]["conclusion"], "결론이다. [E1]")
        self.assertEqual(len(generator.calls), 1)

    def test_partial_answer_is_not_hidden_by_decline_prefix(self):
        generator = StubGenerator(answer(conclusion="근거 부족: 직접 규정한 문단이 없습니다.",
                                         evidence_ids=()))
        result = AccountingQAPipeline(StubRetriever([candidate()]), generator).ask("질문")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["reason"], "evidence_available")
        self.assertEqual(result["answer"]["conclusion"], "근거 부족: 직접 규정한 문단이 없습니다.")

    def test_model_response_is_returned_even_without_evidence(self):
        generator = StubGenerator(answer(conclusion="답변 불가: 관련 문단이 없습니다.",
                                         evidence_ids=()))
        result = AccountingQAPipeline(StubRetriever([candidate()]), generator).ask("질문")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"]["evidence"], [])

    def test_empty_retrieval_skips_generation(self):
        generator = StubGenerator(answer())
        result = AccountingQAPipeline(StubRetriever([]), generator).ask("질문")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "no_evidence_found")
        self.assertEqual(result["answer"]["evidence"], [])
        self.assertEqual(generator.calls, [])

    def test_threshold_filtered_candidates_have_distinct_reason(self):
        generator = StubGenerator(answer())
        result = AccountingQAPipeline(
            StubRetriever([], threshold_filtered_all=True), generator,
        ).ask("질문")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "no_supported_evidence")
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

    def test_image_queries_are_used_only_for_retrieval(self):
        retriever = StubRetriever([candidate()])
        generator = StubGenerator(answer())
        AccountingQAPipeline(retriever, generator).ask(
            "풀어줘",
            semantic_query="금융부채 조건변경의 회계처리",
            keyword_query="금융부채 조건변경 유효이자율",
            image_urls=["data:image/png;base64,YQ=="],
        )

        self.assertEqual(retriever.calls[0][0], "금융부채 조건변경의 회계처리")
        self.assertEqual(
            retriever.calls[0][1]["keyword_query"],
            "금융부채 조건변경 유효이자율",
        )
        self.assertEqual(generator.calls[0][0], "풀어줘")
        self.assertEqual(generator.calls[0][2]["image_urls"], [
            "data:image/png;base64,YQ==",
        ])

    def test_image_only_request_does_not_invent_a_user_question(self):
        retriever = StubRetriever([candidate()])
        generator = StubGenerator(answer())
        AccountingQAPipeline(retriever, generator).ask(
            "",
            semantic_query="금융부채 조건변경의 회계처리",
            keyword_query="금융부채 조건변경",
            image_urls=["data:image/png;base64,YQ=="],
        )

        self.assertEqual(generator.calls[0][0], "")

    def test_generator_evidence_is_passed_through_untouched(self):
        # 출처·원문 조립은 생성기가 끝내므로 파이프라인은 손대지 않는다.
        generated = answer()
        result = AccountingQAPipeline(
            StubRetriever([candidate()]), StubGenerator(generated)
        ).ask("질문")
        self.assertEqual(result["answer"]["evidence"], generated["evidence"])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            AccountingQAPipeline(StubRetriever([]), StubGenerator(answer())).ask("   ")


if __name__ == "__main__":
    unittest.main()
