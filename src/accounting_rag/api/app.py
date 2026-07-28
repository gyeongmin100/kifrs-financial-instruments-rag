from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import secrets
from threading import Lock, Thread
import time
from typing import Any, Callable, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from accounting_rag.api.dependencies import (
    PipelineUnavailable, close_pipeline, get_pipeline, question_with_images,
)
from accounting_rag.api.schemas import (
    AskRequest, AskResponse, HealthResponse, ImageAttachment,
    JobCreatedResponse, JobStatusResponse,
)


logger = logging.getLogger(__name__)
PipelineProvider = Callable[[], Any]
ImageProcessor = Callable[[str, Sequence[ImageAttachment]], str]
STATIC_DIR = Path(__file__).with_name("static")
JOB_TTL_SECONDS = 10 * 60


def _safe_diagnostics(result: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = result.get("retrieval")
    retrieval = retrieval if isinstance(retrieval, Mapping) else {}
    return {
        "retrieval": {
            key: retrieval.get(key)
            for key in ("filters", "seed_count", "sibling_count", "result_count")
        },
        "status": result.get("status"),
        "reason": result.get("reason"),
    }


def _is_service_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, PipelineUnavailable)):
        return True
    return exc.__class__.__module__.split(".", 1)[0] in {"neo4j", "openai"}


def _client_ip(request: Request) -> str:
    # 리버스 프록시(HF Spaces 등) 뒤에서만 X-Forwarded-For를 신뢰한다.
    if os.getenv("TRUST_PROXY_HEADERS") == "1":
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class _RateLimiter:
    """IP당 시간창 요청 제한. limit=0이면 비활성."""

    def __init__(self, limit: int, window_seconds: float = 3600.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        bucket = self._buckets.setdefault(key, deque())
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if not bucket and len(self._buckets) > 10_000:
            self._buckets = {k: v for k, v in self._buckets.items() if v}
            bucket = self._buckets.setdefault(key, deque())
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


class _JobStore:
    """Short-lived process memory for reconnecting to an in-flight answer."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._request_jobs: dict[str, str] = {}
        self._lock = Lock()

    def _prune(self, now: float) -> None:
        expired = [
            job_id for job_id, job in self._jobs.items()
            if now - float(job["updated_at"]) > JOB_TTL_SECONDS
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id, None)
            if job and job.get("request_id"):
                self._request_jobs.pop(str(job["request_id"]), None)

    def create(
        self,
        request_id: str | None = None,
        before_create: Callable[[], None] | None = None,
    ) -> tuple[str, bool]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            if request_id:
                existing = self._request_jobs.get(request_id)
                if existing is not None:
                    return existing, False
            if before_create is not None:
                before_create()
            job_id = secrets.token_urlsafe(24)
            self._jobs[job_id] = {
                "status": "pending", "updated_at": now,
                "request_id": request_id,
            }
            if request_id:
                self._request_jobs[request_id] = job_id
        return job_id, True

    def finish(self, job_id: str, result: AskResponse) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id] = {
                    "status": "complete", "result": result,
                    "request_id": self._jobs[job_id].get("request_id"),
                    "updated_at": time.monotonic(),
                }

    def fail(self, job_id: str, detail: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id] = {
                    "status": "error", "detail": detail,
                    "request_id": self._jobs[job_id].get("request_id"),
                    "updated_at": time.monotonic(),
                }

    def get(self, job_id: str) -> dict[str, Any] | None:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {key: value for key, value in job.items() if key != "request_id"}

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job and job.get("request_id"):
                self._request_jobs.pop(str(job["request_id"]), None)


def create_app(
    pipeline_provider: PipelineProvider | None = None,
    image_processor: ImageProcessor = question_with_images,
) -> FastAPI:
    provider = pipeline_provider or get_pipeline
    owns_default = pipeline_provider is None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_default:
            close_pipeline()

    app = FastAPI(title="Accounting RAG API", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["POST", "GET", "PATCH", "DELETE"],
            allow_headers=["Content-Type", "Accept", "X-Review-Token"],
        )

    rate_limiter = _RateLimiter(int(os.getenv("ASK_RATE_LIMIT_PER_HOUR", "0")))
    jobs = _JobStore()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    def run_request(request: AskRequest) -> AskResponse:
        try:
            pipeline = provider()
            prepared_question = image_processor(request.question, request.images)
            result = pipeline.ask(
                prepared_question,
                standard_id=request.standard_id,
                zone=request.zone,
                top_k=request.top_k,
            )
        except ValueError as exc:
            logger.info("ask request rejected: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="요청 값을 처리할 수 없습니다.") from exc
        except Exception as exc:
            status_code = 503 if _is_service_error(exc) else 500
            logger.warning("ask failed: %s", type(exc).__name__)
            detail = (
                "질의 처리 서비스에 일시적으로 연결할 수 없습니다."
                if status_code == 503
                else "질의 처리 중 내부 오류가 발생했습니다."
            )
            raise HTTPException(status_code=status_code, detail=detail) from exc

        try:
            answer = result["answer"]
            return AskResponse(
                status=str(result["status"]),
                reason=str(result.get("reason") or _answer_reason(result)),
                conclusion=answer["conclusion"],
                reasoning=answer["reasoning"],
                evidence=answer["evidence"],
                diagnostics=_safe_diagnostics(result) if request.debug else None,
            )
        except Exception as exc:
            logger.warning("invalid pipeline response: %s", type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="질의 처리 중 내부 오류가 발생했습니다.",
            ) from exc

    def check_rate_limit(http_request: Request) -> None:
        if not rate_limiter.allow(_client_ip(http_request)):
            raise HTTPException(
                status_code=429,
                detail="요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
            )

    def _answer_reason(result: Mapping[str, Any]) -> str:
        status = str(result.get("status") or "")
        if status == "answered":
            return "sufficient"
        if status == "generation_failed":
            return "generation_failed"
        sufficiency = result.get("sufficiency")
        if isinstance(sufficiency, Mapping) and sufficiency.get("status"):
            return str(sufficiency["status"])
        return status or "unknown"

    @app.post(
        "/v1/ask",
        response_model=AskResponse,
        response_model_exclude_none=True,
    )
    def ask(request: AskRequest, http_request: Request) -> AskResponse:
        check_rate_limit(http_request)
        return run_request(request)

    @app.post("/v1/jobs", response_model=JobCreatedResponse, status_code=202)
    def create_job(request: AskRequest, http_request: Request) -> JobCreatedResponse:
        request_id = str(request.request_id) if request.request_id else None
        job_id, created = jobs.create(
            request_id, lambda: check_rate_limit(http_request),
        )
        if not created:
            return JobCreatedResponse(job_id=job_id)

        def worker() -> None:
            try:
                jobs.finish(job_id, run_request(request))
            except HTTPException as exc:
                jobs.fail(job_id, str(exc.detail))
            except Exception as exc:  # defensive boundary around the worker thread
                logger.warning("answer job failed: %s", type(exc).__name__)
                jobs.fail(job_id, "질의 처리 중 내부 오류가 발생했습니다.")

        Thread(target=worker, name=f"answer-{job_id[:8]}", daemon=True).start()
        return JobCreatedResponse(job_id=job_id)

    @app.get(
        "/v1/jobs/{job_id}", response_model=JobStatusResponse,
        response_model_exclude_none=True,
    )
    def get_job(job_id: str) -> JobStatusResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="대기 중인 요청을 찾을 수 없습니다.")
        return JobStatusResponse(**job)

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        jobs.delete(job_id)

    return app


app = create_app()
