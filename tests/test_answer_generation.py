import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.generation.answer import AnswerConfig, OpenAIAnswerGenerator


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


def candidate(chunk_id="C1", text="기대신용손실을 측정한다.", citation="제1109호 5.5.17"):
    return {"chunk_id": chunk_id, "text": text, "citation_label": citation}


def valid_payload():
    return {
        "conclusion": "보고기간 말에 기대신용손실을 측정한다. [E1]",
        "reasoning": ["해당 문단은 측정 원칙을 직접 규정한다. [E1]"],
        "evidence": [{
            "evidence_id": "E1",
            "citation": "제1109호 5.5.17",
            "statement": "기대신용손실 측정 원칙",
        }],
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
        sent = json.loads(call["input"][1]["content"])
        self.assertEqual(sent["evidence"][0]["source_id"], "C1")

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

    def test_unknown_duplicate_id_or_wrong_citation_falls_back(self):
        invalid = []
        unknown = valid_payload()
        unknown["evidence"][0]["evidence_id"] = "E9"
        invalid.append(unknown)
        duplicate = valid_payload()
        duplicate["evidence"].append(dict(duplicate["evidence"][0]))
        invalid.append(duplicate)
        wrong_citation = valid_payload()
        wrong_citation["evidence"][0]["citation"] = "invented"
        invalid.append(wrong_citation)
        for payload in invalid:
            with self.subTest(payload=payload):
                result = OpenAIAnswerGenerator(
                    SimpleNamespace(responses=FakeResponses(payload)), model="model"
                ).generate("질문", [candidate()])
                self.assertEqual(result["evidence"], [])
                self.assertTrue(result["conclusion"].startswith("근거 부족:"))

    def test_uncited_narrative_falls_back(self):
        uncited = valid_payload()
        uncited["conclusion"] = "기대신용손실을 측정한다."
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(uncited)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual(result["evidence"], [])

    def test_one_reasoning_step_may_contain_explanation_before_its_citation(self):
        payload = valid_payload()
        payload["reasoning"] = ["①은 정의와 일치한다. ②는 지급시기를 무시하므로 틀리다. [E1]"]
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(payload)), model="model"
        ).generate("질문", [candidate()])
        self.assertEqual(result, payload)

    def test_unused_evidence_falls_back(self):

        unused = valid_payload()
        unused["evidence"].append({
            "evidence_id": "E2", "citation": "문단 2", "statement": "보조 근거"
        })
        result = OpenAIAnswerGenerator(
            SimpleNamespace(responses=FakeResponses(unused)), model="model"
        ).generate("질문", [candidate(), candidate("C2", citation="문단 2")])
        self.assertEqual(result["evidence"], [])

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

    def test_api_failure_and_empty_candidates_return_safe_fallback(self):
        responses = FakeResponses(error=RuntimeError("temporary"))
        generator = OpenAIAnswerGenerator(SimpleNamespace(responses=responses), model="model")
        self.assertEqual(generator.generate("질문", [candidate()])["evidence"], [])
        self.assertEqual(generator.generate("질문", [])["evidence"], [])
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
