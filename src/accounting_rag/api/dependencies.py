from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from pydantic import BaseModel

from accounting_rag.api.schemas import ImageAttachment


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PipelineUnavailable(RuntimeError):
    """Raised when the configured OpenAI/Neo4j pipeline cannot be created."""


@dataclass(frozen=True)
class ImageSearchQuery:
    semantic_query: str
    keywords: tuple[str, ...]

    @property
    def keyword_query(self) -> str:
        return " ".join(self.keywords)


class ImageSearchOutput(BaseModel):
    semantic_query: str
    keywords: list[str]


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
) -> ImageSearchQuery:
    """Create separate semantic and keyword searches from attached images."""
    if not images:
        clean_question = question.strip()
        return ImageSearchQuery(clean_question, (clean_question,))

    from openai import OpenAI

    content: list[dict[str, str]] = [{
        "type": "input_text",
        "text": f"""당신은 이미지를 분석하여 검색 정보를 만드는 K-IFRS 금융상품 회계 전문가다.

사용자의 질문과 첨부 이미지를 분석하고, 관련 자료를 벡터검색과 키워드 검색으로 찾기 위한 검색 정보를 작성한다.

규칙:
1. 거래의 실질, 금융상품 유형 및 핵심 회계 쟁점을 파악한다.
2. 문제의 정답을 계산하거나 제시하지 않는다.
3. semantic_query에는 핵심 회계 쟁점을 관련 자료에서 사용될 만한 회계 용어로 설명하는 완전한 자연어 문장을 작성한다.
4. 문제 문장을 그대로 옮기지 않는다.
5. keywords에는 키워드 검색에 사용할 핵심 회계 용어만 작성한다.
6. 금액, 날짜, 회사명 등 검색에 불필요한 사실은 제외하되, 회계 판단에 영향을 미치는 계약조건과 거래 특성은 포함한다.
7. 입력에서 확인되지 않은 사실이나 회계 쟁점을 추가하지 않는다.
8. 이미지가 불명확한 경우 확인할 수 있는 범위에서만 작성한다.
9. 문제의 정답이나 계산 결과를 검색 정보에 포함하지 않는다.
10. JSON 외의 내용은 출력하지 않는다.

사용자의 질문: {question or '(없음)'}""",
    }]
    content.extend({
        "type": "input_image",
        "image_url": f"data:{image.mime_type};base64,{image.data}",
        "detail": "high",
    } for image in images)
    response = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).responses.parse(
        model=os.environ["OPENAI_CHAT_MODEL"],
        input=[{"role": "user", "content": content}],
        text_format=ImageSearchOutput,
        max_output_tokens=8_000,
        store=False,
    )
    if response.status != "completed" or response.output_parsed is None:
        raise PipelineUnavailable("image search analysis did not complete")

    semantic_query = response.output_parsed.semantic_query.strip()
    keywords = tuple(dict.fromkeys(
        keyword.strip() for keyword in response.output_parsed.keywords
        if keyword.strip()
    ))
    if not semantic_query or not keywords:
        raise PipelineUnavailable("image search analysis returned an empty query")
    return ImageSearchQuery(semantic_query[:6_000], keywords)
