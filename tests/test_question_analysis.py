import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.query.analysis import (
    OpenAIQuestionAnalyzer, QuestionAnalysisConfig,
    extract_explicit_paragraphs, extract_explicit_standards,
)


class FakeResponses:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=json.dumps(self.payload, ensure_ascii=False))


def payload(**changes):
    result = {
        "requested_standard_ids": ["1109"],
        "requested_paragraphs": ["3.2.3~3.2.9"],
        "concepts": ["금융자산 제거"],
        "subquestions": ["해당 문단의 요구사항"],
        "search_query": "K-IFRS 1109 금융자산 제거 3.2.3 3.2.9",
    }
    result.update(changes)
    return result


class QuestionAnalysisTests(unittest.TestCase):
    def analyzer(self, responses):
        return OpenAIQuestionAnalyzer(SimpleNamespace(responses=responses), model="model")

    def test_deterministic_extractors(self):
        self.assertEqual(
            extract_explicit_standards("K-IFRS 제1109호와 IAS 32, IFRS 7"),
            ("1109", "1032", "1107"),
        )
        self.assertEqual(
            extract_explicit_paragraphs("문단 3.2.3-3.2.9와 문단 B5.5.51, 35H; 손실률 10~20%"),
            ("3.2.3~3.2.9", "B5.5.51", "35H"),
        )

    def test_config_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.yaml"
            path.write_text("query_analysis:\n  allowed_standard_ids: ['1109']\n  max_output_tokens: 900\n", encoding="utf-8")
            config = QuestionAnalysisConfig.from_yaml(path)
        self.assertEqual(config.allowed_standard_ids, ("1109",))
        self.assertEqual(config.max_output_tokens, 900)

    def test_strict_responses_api_and_valid_result(self):
        responses = FakeResponses(payload())
        result = self.analyzer(responses).analyze("K-IFRS 제1109호 문단 3.2.3~3.2.9를 설명해줘")
        self.assertEqual(result.requested_paragraphs, ("3.2.3~3.2.9",))
        call = responses.calls[0]
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertTrue(call["text"]["format"]["strict"])
        self.assertFalse(call["store"])

    def test_changed_constraint_and_invalid_paragraph_fall_back(self):
        cases = [
            payload(requested_standard_ids=["1032"]),
            payload(requested_paragraphs=["bad paragraph"]),
        ]
        for response in cases:
            result = self.analyzer(FakeResponses(response)).analyze("제1109호 문단 3.2.3~3.2.9를 설명해줘")
            self.assertEqual(result.requested_standard_ids, ("1109",))
            self.assertEqual(result.requested_paragraphs, ("3.2.3~3.2.9",))
            self.assertEqual(result.concepts, ())

    def test_unknown_explicit_standard_rejected_before_api(self):
        responses = FakeResponses(payload())
        with self.assertRaises(ValueError):
            self.analyzer(responses).analyze("기업회계기준서 제1115호를 설명해줘")
        self.assertEqual(responses.calls, [])

    def test_response_format_classification_field_is_rejected(self):
        classified = payload(question_type="multiple_choice")
        result = self.analyzer(FakeResponses(classified)).analyze("다음 중 옳은 것은?")
        self.assertEqual(result.search_query, "다음 중 옳은 것은?")
        self.assertFalse(hasattr(result, "question_type"))

    def test_api_failure_fallback_preserves_original_query_and_constraints(self):
        question = "제1039호 문단 42E의 금액을 계산해줘"
        result = self.analyzer(FakeResponses(error=RuntimeError("temporary"))).analyze(question)
        self.assertEqual(result.requested_standard_ids, ("1039",))
        self.assertEqual(result.requested_paragraphs, ("42E",))
        self.assertEqual(result.search_query, question)


if __name__ == "__main__":
    unittest.main()
