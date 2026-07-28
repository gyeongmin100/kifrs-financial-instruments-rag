from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from accounting_rag.api.schemas import ImageAttachment


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PipelineUnavailable(RuntimeError):
    """Raised when the configured OpenAI/Neo4j pipeline cannot be created."""


_pipeline: Any | None = None
_driver: Any | None = None
_lock = Lock()


def _build_pipeline() -> tuple[Any, Any]:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    from openai import OpenAI

    from accounting_rag.generation.answer import AnswerConfig, OpenAIAnswerGenerator
    from accounting_rag.qa_pipeline import AccountingQAPipeline
    from accounting_rag.retrieval.hybrid import HybridConfig, HybridRetriever
    from accounting_rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

    load_dotenv(PROJECT_ROOT / ".env")
    required = (
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
        "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL", "OPENAI_CHAT_MODEL",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise PipelineUnavailable("required service configuration is missing")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        retrieval_config = PROJECT_ROOT / "config" / "retrieval.yaml"
        retriever = RetrievalPipeline(
            HybridRetriever(
                driver, client,
                database=os.environ["NEO4J_DATABASE"],
                embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
                config=HybridConfig.from_yaml(retrieval_config),
            ),
            driver,
            database=os.environ["NEO4J_DATABASE"],
            config=RetrievalConfig.from_yaml(retrieval_config),
        )
        pipeline = AccountingQAPipeline(
            retriever,
            OpenAIAnswerGenerator(
                client,
                model=os.environ["OPENAI_CHAT_MODEL"],
                config=AnswerConfig.from_yaml(PROJECT_ROOT / "config" / "answering.yaml"),
            ),
        )
        return pipeline, driver
    except Exception:
        driver.close()
        raise


def get_pipeline() -> Any:
    """Return one lazily created process-local pipeline."""
    global _pipeline, _driver
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        try:
            _pipeline, _driver = _build_pipeline()
        except PipelineUnavailable:
            raise
        except Exception as exc:
            raise PipelineUnavailable("pipeline initialization failed") from exc
    return _pipeline


def close_pipeline() -> None:
    """Release the default Neo4j driver; injected providers are caller-owned."""
    global _pipeline, _driver
    with _lock:
        driver, _driver, _pipeline = _driver, None, None
    if driver is not None:
        driver.close()


def question_with_images(
    question: str, images: Sequence[ImageAttachment],
) -> str:
    """Turn attached images into retrieval text without persisting them."""
    if not images:
        return question

    from openai import OpenAI

    content: list[dict[str, str]] = [{
        "type": "input_text",
        "text": (
            "첨부 이미지를 회계 질의에 사용할 수 있도록 분석하세요. 이미지 속 문구와 숫자를 "
            "가능한 정확히 옮기고, 표라면 행과 열의 관계를 보존하세요. 설명이나 추측을 덧붙이지 "
            f"말고 검색 가능한 한국어 텍스트만 반환하세요. 사용자의 질문: {question or '(없음)'}"
        ),
    }]
    content.extend({
        "type": "input_image",
        "image_url": f"data:{image.mime_type};base64,{image.data}",
        "detail": "high",
    } for image in images)
    response = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).responses.create(
        model=os.environ["OPENAI_CHAT_MODEL"],
        input=[{"role": "user", "content": content}],
        max_output_tokens=2_000,
        store=False,
    )
    extracted = response.output_text.strip()
    combined = f"{question}\n\n첨부 이미지 내용:\n{extracted}" if question else extracted
    return combined[:12_000]
