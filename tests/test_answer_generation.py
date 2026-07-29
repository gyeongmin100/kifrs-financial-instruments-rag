import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.generation.answer import (
    AnswerConfig, AnswerGenerationError, OpenAIAnswerGenerator,
)


class FakeResponses:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=json.dumps(self.payload, ensure_ascii=False))


def candidate(chunk_id="C1", text="기대신용손실을 측정한다.", citation="제1109호 5.5.17",
              **metadata):
    return {"chunk_id": chunk_id, "text": text, "citation_label": citation, **metadata}


def valid_payload():
    # 모델은 인용한 evidence_id만 돌려준다. 출처와 원문은 생성기가 목록에서 채운다.
    return {
        "conclusion": "보고기간 말에 기대신용손실을 측정한다. [E1]",
        "reasoning": ["해당 문단은 측정 원칙을 직접 규정한다. [E1]"],
        "evidence": ["E1"],
    }


class AnswerGenerationTests(unittest.TestCase):
    def test_generates_three_section_structured_answer(self):
        responses = FakeResponses(valid_payload())
        generator = OpenAIAnswerGenerator(
            SimpleNamespace(responses=responses), model="answer-model"
        )
        result = generator.generate("어떻게 측정하는가?", [candidate()])
        self.assertEqual(set(result), {"conclusion", "reasoning", "evidence"})
        self.assertEqual(result["evidence"][0]["evidence_id"], "E1")
        call = responses.calls[0]
        self.assertEqual(call["model"], "answer-model")
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertFalse(call["store"])
        system_prompt = call["input"][0]["content"]
        self.assertIn("evidence를 회계기준 판단의 최우선 근거로 사용한다", system_prompt)
        self.assertIn("사실, 숫자, 표, 보기 등의 전제조건은 입력정보", system_prompt)
        self.assertIn("일반 회계 지식, 계산과 논리적 추론", system_prompt)
        self.assertIn("일부만 답할 수 있어도 확인 가능한 부분은 답하고", system_prompt)
        sent = json.loads(call["input"][1]["content"])
        self.assertEqual(sent["evidence"][0]["source_id"], "C1")

    def test_original_images_are_included_with_user_question(self):
        responses = FakeResponses(valid_payload())
        generator = OpenAIAnswerGenerator(
            SimpleNamespace(responses=responses), model="answer-model"
        )
        generator.generate(
            "질문",
            [candidate()],
            image_urls=["data:image/png;base64,YQ=="],
        )

        content = responses.calls[0]["input"][1]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn("질문", content[0]["text"])
        self.assertEqual(content[1], {
            "type": "input_image",
            "image_url": "data:image/png;base64,YQ==",
            "detail": "original",
        })
        self.assertIn("원본 이미지", responses.calls[0]["input"][0]["content"])

    def test_image_only_input_omits_an_invented_question(self):
        responses = FakeResponses(valid_payload())
        generator = OpenAIAnswerGenerator(
            SimpleNamespace(responses=responses), model="answer-model"
        )
        generator.generate(
            "", [candidate()], image_urls=["data:image/png;base64,YQ=="],
        )

        content = responses.calls[0]["input"][1]["content"]
        sent = json.loads(content[0]["text"])
        self.assertNotIn("question", sent)
        self.assertIn("evidence", sent)

    def test_limits_candidate_count_individual_and_total_text(self):
        payload = valid_payload()
        responses = FakeResponses(payload)
        config = AnswerConfig(max_candidates=2, max_candidate_chars=5, max_context_chars=8)
        generator = OpenAIAnswerGenerator(
            SimpleNamespace(responses=responses), model="model", config=config
        )
        generator.generate("질문", [
            candidate("C1", "가" * 20), candidate("C2", "나" * 20, "문단 2"),
            candidate("C3", "다" * 20, "문단 3"),
        ])
        offered = json.loads(responses.calls[0]["input"][1]["content"])["evidence"]
        self.assertEqual([len(item["statement"]) for item in offered], [5, 3])
        self.assertEqual([item["evidence_id"] for item in offered], ["E1", "E2"])

    def test_citation_and_statement_come_from_the_catalog(self):
        # 모델은 evidence_id만 말하므로 출처와 원문을 지어낼 통로 자체가 없다.
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(valid_payload())), model="model"
        ).generate("질문", [candidate(pdf_page_start=12, candidate_source="hybrid")])
        self.assertEqual(result["evidence"], [{
            "evidence_id": "E1",
            "citation": "제1109호 5.5.17",
            "statement": "기대신용손실을 측정한다.",
            "source_id": "C1",
            "pdf_page_start": 12,
            "candidate_source": "hybrid",
        }])

    def test_unknown_and_duplicate_ids_are_dropped_without_discarding_answer(self):
        payload = valid_payload()
        payload["evidence"] = ["E9", "E1", "E1"]
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(payload)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1"])
        self.assertEqual(result["conclusion"], payload["conclusion"])

    def test_narrative_without_a_marker_is_kept(self):
        uncited = valid_payload()
        uncited["conclusion"] = "기대신용손실을 측정한다."
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(uncited)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual(result["conclusion"], "기대신용손실을 측정한다.")
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1"])

    def test_marker_only_in_the_text_still_produces_an_evidence_card(self):
        # 목록에서 빠졌더라도 본문이 인용했으면 카드를 만든다.
        payload = valid_payload()
        payload["evidence"] = []
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(payload)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1"])

    def test_one_reasoning_step_may_contain_explanation_before_its_citation(self):
        payload = valid_payload()
        payload["reasoning"] = ["①은 정의와 일치한다. ②는 지급시기를 무시하므로 틀리다. [E1]"]
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(payload)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual(result["reasoning"], payload["reasoning"])
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1"])

    def test_declared_but_uncited_evidence_is_kept(self):
        payload = valid_payload()
        payload["evidence"] = ["E1", "E2"]
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(payload)), model="model"
        ).generate("질문", [candidate(), candidate("C2", citation="문단 2")])
        self.assertEqual([item["evidence_id"] for item in result["evidence"]], ["E1", "E2"])

    def test_insufficient_answer_without_citation_is_allowed(self):
        payload = {
            "conclusion": "근거 부족: 사실관계가 없어 판단할 수 없습니다.",
            "reasoning": ["근거 부족: 계약조건에 관한 정보가 필요합니다."],
            "evidence": [],
        }
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(payload)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual(result, payload)

    def test_api_failure_and_empty_candidates_raise_generation_error(self):
        responses = FakeResponses(error=RuntimeError("temporary"))
        generator = OpenAIAnswerGenerator(SimpleNamespace(responses=responses), model="model")
        with self.assertRaises(AnswerGenerationError):
            generator.generate("질문", [candidate()])
        with self.assertRaises(AnswerGenerationError):
            generator.generate("질문", [])
        self.assertEqual(len(responses.calls), 1)

    def test_input_validation_rejects_empty_long_and_duplicate_inputs(self):
        generator = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(valid_payload())), model="model",
            config=AnswerConfig(max_question_chars=3),
        )
        with self.assertRaises(ValueError):
            generator.generate(" ", [candidate()])
        with self.assertRaises(ValueError):
            generator.generate("질문길이", [candidate()])
        with self.assertRaises(ValueError):
            generator.generate("질문", [candidate(), candidate()])


if __name__ == "__main__":
    unittest.main()
