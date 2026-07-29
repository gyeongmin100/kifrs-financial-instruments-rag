import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from accounting_rag.api.dependencies import (
    ImageSearchOutput, PipelineUnavailable, question_with_images,
)
from accounting_rag.api.schemas import ImageAttachment


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class ImageSearchTests(unittest.TestCase):
    def test_image_analysis_returns_separate_queries_with_8000_token_limit(self):
        responses = FakeResponses(SimpleNamespace(
            status="completed",
            output_parsed=ImageSearchOutput(
                semantic_query="금융부채 조건변경의 회계처리",
                keywords=["금융부채", "조건변경", "금융부채"],
            ),
        ))
        image = ImageAttachment(
            name="question.png", mime_type="image/png", data="YQ==",
        )
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key", "OPENAI_CHAT_MODEL": "test-model",
        }), patch("openai.OpenAI", return_value=SimpleNamespace(responses=responses)):
            result = question_with_images("풀어줘", [image])

        self.assertEqual(result.semantic_query, "금융부채 조건변경의 회계처리")
        self.assertEqual(result.keyword_query, "금융부채 조건변경")
        call = responses.calls[0]
        self.assertEqual(call["max_output_tokens"], 8_000)
        self.assertIs(call["text_format"], ImageSearchOutput)
        self.assertIn("문제의 정답을 계산하거나 제시하지 않는다", call["input"][0]["content"][0]["text"])
        self.assertEqual(call["input"][0]["content"][1]["type"], "input_image")

    def test_incomplete_image_analysis_is_rejected(self):
        responses = FakeResponses(SimpleNamespace(
            status="incomplete", output_parsed=None,
        ))
        image = ImageAttachment(
            name="question.png", mime_type="image/png", data="YQ==",
        )
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key", "OPENAI_CHAT_MODEL": "test-model",
        }), patch("openai.OpenAI", return_value=SimpleNamespace(responses=responses)):
            with self.assertRaises(PipelineUnavailable):
                question_with_images("", [image])


if __name__ == "__main__":
    unittest.main()
