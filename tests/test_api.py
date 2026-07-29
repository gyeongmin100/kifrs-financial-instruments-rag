import unittest
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from fastapi.testclient import TestClient
except ImportError:  # dependencies may not be installed in a parser-only environment
    TestClient = None

from accounting_rag.api.dependencies import PipelineUnavailable


class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result or {
            "status": "answered",
            "reason": "evidence_available",
            "answer": {
                "conclusion": "결론 [E1]",
                "reasoning": ["판단 [E1]"],
                "evidence": [{
                    "evidence_id": "E1", "citation": "제1109호 5.5.17",
                    "statement": "근거", "source_id": "KIFRS1109-5.5.17",
                    "pdf_page_start": 41, "pdf_page_end": 42,
                    "graph_path": ["seed", "REFERS_TO", "target"],
                    "graph_hop": 1, "candidate_source": "graph",
                }],
            },
            "retrieval": {
                "question": "비공개",
                "filters": {"standard_id": "1109", "zone": "standard_body"},
                "seed_count": 10,
                "graph_evidence_count": 20,
                "candidate_count": 30,
                "result_count": 10,
                "results": [{"text": "원문 전체"}],
                "attempts": [{"attempt": 1, "search_question": "비공개", "result_count": 10, "sufficiency_status": "sufficient"}],
            },
            "sufficiency": {"status": "sufficient", "sufficient": True, "semantic": {"confidence": 0.9, "missing_aspects": []}},
            "citation_verification": {"valid": True},
            "analysis": {"concepts": ["기대신용손실"]},
        }
        self.error = error
        self.calls = []

    def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        if self.error:
            raise self.error
        return self.result


@unittest.skipIf(TestClient is None, "FastAPI test dependencies are not installed")
class ApiTests(unittest.TestCase):
    def client(self, pipeline=None, provider=None):
        from accounting_rag.api.app import create_app
        return TestClient(create_app(provider or (lambda: pipeline)))

    def test_health_does_not_initialize_pipeline(self):
        calls = []
        with self.client(provider=lambda: calls.append(True)) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "accounting-rag-api"})
        self.assertEqual(calls, [])

    def test_root_serves_ui_without_initializing_pipeline(self):
        calls = []
        with self.client(provider=lambda: calls.append(True)) as client:
            response = client.get("/")
            stylesheet = client.get("/static/styles.css")
            favicon = client.get("/favicon.ico")
            static_favicon = client.get("/static/favicon.png")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Accountant", response.text)
        # 정적 호스팅(Cloudflare Pages)에는 /favicon.ico 라우트가 없으므로 실제 파일을 가리킨다.
        self.assertIn('href="/static/favicon.png"', response.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertEqual(favicon.status_code, 200)
        self.assertEqual(favicon.headers["content-type"], "image/png")
        self.assertTrue(favicon.content.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(static_favicon.status_code, 200)
        self.assertTrue(static_favicon.content.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertLess(len(static_favicon.content), 100_000)
        self.assertEqual(calls, [])

    def test_ask_returns_public_answer_and_passes_filters(self):
        pipeline = FakePipeline()
        with self.client(pipeline) as client:
            response = client.post("/v1/ask", json={
                "question": " 기대신용손실은? ", "standard_id": "1109",
                "zone": "standard_body", "top_k": 7,
            })
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("diagnostics", response.json())
        self.assertEqual(response.json()["reason"], "evidence_available")
        self.assertEqual(response.json()["conclusion"], "결론 [E1]")
        evidence = response.json()["evidence"][0]
        self.assertEqual(evidence["source_id"], "KIFRS1109-5.5.17")
        self.assertEqual(evidence["pdf_page_start"], 41)
        self.assertEqual(evidence["graph_hop"], 1)
        self.assertEqual(evidence["candidate_source"], "graph")
        self.assertEqual(pipeline.calls, [("기대신용손실은?", {
            "standard_id": "1109", "zone": "standard_body", "top_k": 7,
        })])

    def test_debug_has_diagnostics_but_not_question_or_source_text(self):
        with self.client(FakePipeline()) as client:
            payload = client.post("/v1/ask", json={"question": "질문", "debug": True}).json()
        diagnostic_text = str(payload["diagnostics"])
        self.assertNotIn("원문 전체", diagnostic_text)
        self.assertNotIn("search_question", diagnostic_text)
        self.assertEqual(payload["diagnostics"]["retrieval"]["seed_count"], 10)

    def test_schema_validation_rejects_invalid_values(self):
        invalid = [
            {"question": "   "},
            {"question": "질문", "standard_id": "1110"},
            {"question": "질문", "zone": "unknown"},
            {"question": "질문", "top_k": 0},
            {"question": "질문", "unexpected": True},
        ]
        with self.client(FakePipeline()) as client:
            self.assertTrue(all(client.post("/v1/ask", json=item).status_code == 422 for item in invalid))

    def test_question_length_limit_is_2000_characters(self):
        pipeline = FakePipeline()
        with self.client(pipeline) as client:
            accepted = client.post("/v1/ask", json={"question": "가" * 2_000})
            rejected = client.post("/v1/ask", json={"question": "가" * 2_001})
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(pipeline.calls[0][0], "가" * 2_000)

    def test_pipeline_value_error_is_422(self):
        with self.client(FakePipeline(error=ValueError("do not expose"))) as client:
            response = client.post("/v1/ask", json={"question": "질문"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("do not expose", response.text)

    def test_unavailable_is_503_and_unexpected_is_500(self):
        with self.client(provider=lambda: (_ for _ in ()).throw(PipelineUnavailable("secret"))) as client:
            unavailable = client.post("/v1/ask", json={"question": "질문"})
        with self.client(FakePipeline(error=RuntimeError("secret"))) as client:
            failure = client.post("/v1/ask", json={"question": "질문"})
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(failure.status_code, 500)
        self.assertNotIn("secret", unavailable.text + failure.text)

    def test_malformed_pipeline_response_is_500(self):
        pipeline = FakePipeline(result={"status": "answered", "answer": {"conclusion": 1}})
        with self.client(pipeline) as client:
            response = client.post("/v1/ask", json={"question": "질문"})
        self.assertEqual(response.status_code, 500)

    def test_short_lived_job_can_be_polled_and_deleted(self):
        pipeline = FakePipeline()
        with self.client(pipeline) as client:
            created = client.post("/v1/jobs", json={"question": "질문"})
            self.assertEqual(created.status_code, 202)
            job_id = created.json()["job_id"]
            for _ in range(50):
                response = client.get(f"/v1/jobs/{job_id}")
                if response.json()["status"] != "pending":
                    break
                time.sleep(0.01)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "complete")
            self.assertEqual(response.json()["result"]["reason"], "evidence_available")
            self.assertEqual(response.json()["result"]["conclusion"], "결론 [E1]")
            self.assertEqual(client.delete(f"/v1/jobs/{job_id}").status_code, 204)
            self.assertEqual(client.get(f"/v1/jobs/{job_id}").status_code, 404)

    def test_insufficient_response_exposes_internal_reason(self):
        for reason in ("self_declined", "no_evidence_found", "no_supported_evidence"):
            with self.subTest(reason=reason):
                result = FakePipeline().result
                result["status"] = "insufficient"
                result["reason"] = reason
                result["answer"] = {
                    "conclusion": "근거 부족: 답할 수 없습니다.", "reasoning": [], "evidence": [],
                }
                with self.client(FakePipeline(result=result)) as client:
                    response = client.post("/v1/ask", json={"question": "question"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "insufficient")
                self.assertEqual(response.json()["reason"], reason)

    def test_request_id_reuses_job_and_starts_one_worker_under_concurrency(self):
        pipeline = FakePipeline()
        release = Event()

        def blocked_ask(question, **kwargs):
            pipeline.calls.append((question, kwargs))
            release.wait(timeout=2)
            if pipeline.error:
                raise pipeline.error
            return pipeline.result

        pipeline.ask = blocked_ask
        payload = {"question": "question", "request_id": "12345678-1234-4234-8234-123456789abc"}
        with self.client(pipeline) as client:
            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _: client.post("/v1/jobs", json=payload), range(2)))
            self.assertTrue(all(response.status_code == 202 for response in responses))
            self.assertEqual(responses[0].json()["job_id"], responses[1].json()["job_id"])
            for _ in range(50):
                if pipeline.calls:
                    break
                time.sleep(0.01)
            time.sleep(0.05)
            self.assertEqual(len(pipeline.calls), 1)
            release.set()

    def test_request_id_is_optional_and_must_be_uuid(self):
        with self.client(FakePipeline()) as client:
            first = client.post("/v1/jobs", json={"question": "question"})
            second = client.post("/v1/jobs", json={"question": "question"})
            invalid = client.post("/v1/jobs", json={"question": "question", "request_id": "not-a-uuid"})
        self.assertNotEqual(first.json()["job_id"], second.json()["job_id"])
        self.assertEqual(invalid.status_code, 422)

    def test_request_id_retry_does_not_consume_rate_limit_twice(self):
        payload = {
            "question": "question",
            "request_id": "12345678-1234-4234-8234-123456789abc",
        }
        with patch.dict("os.environ", {"ASK_RATE_LIMIT_PER_HOUR": "1"}):
            with self.client(FakePipeline()) as client:
                first = client.post("/v1/jobs", json=payload)
                retry = client.post("/v1/jobs", json=payload)
                other = client.post("/v1/jobs", json={
                    "question": "other",
                    "request_id": "abcdefab-1234-4234-8234-123456789abc",
                })
        self.assertEqual(first.status_code, 202)
        self.assertEqual(retry.status_code, 202)
        self.assertEqual(first.json()["job_id"], retry.json()["job_id"])
        self.assertEqual(other.status_code, 429)

    def test_image_creates_separate_dense_and_sparse_queries(self):
        pipeline = FakePipeline()
        seen = []

        def image_processor(question, images):
            seen.append((question, images))
            from accounting_rag.api.dependencies import ImageSearchQuery
            return ImageSearchQuery(
                "금융부채 조건변경의 회계처리",
                ("금융부채", "조건변경", "유효이자율"),
            )

        from accounting_rag.api.app import create_app
        with TestClient(create_app(lambda: pipeline, image_processor=image_processor)) as client:
            response = client.post("/v1/ask", json={
                "question": "", "images": [{
                    "name": "question.png", "mime_type": "image/png", "data": "YQ==",
                }],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(seen[0][1]), 1)
        self.assertEqual(pipeline.calls[0][0], "")
        self.assertEqual(
            pipeline.calls[0][1]["semantic_query"], "금융부채 조건변경의 회계처리",
        )
        self.assertEqual(
            pipeline.calls[0][1]["keyword_query"], "금융부채 조건변경 유효이자율",
        )
        self.assertEqual(pipeline.calls[0][1]["image_urls"], [
            "data:image/png;base64,YQ==",
        ])


if __name__ == "__main__":
    unittest.main()
