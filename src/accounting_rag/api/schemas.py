from __future__ import annotations

import base64
import binascii
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StandardId = Literal["1032", "1039", "1107", "1109"]
Zone = Literal[
    "standard_body",
    "appendix_definitions",
    "application_guidance",
    "application_examples",
    "implementation_guidance",
    "basis_for_conclusions",
    "dissenting_opinion",
    "front_matter",
    "committee_resolution",
    "amendment_history",
    "ifrs_comparison",
]
AnswerStatus = Literal["answered", "insufficient"]
AnswerReason = Literal[
    "sufficient",
    # 답변 생성기가 제공된 근거로는 답할 수 없다고 스스로 밝힌 경우.
    "self_declined",
    # 검색이 아무 문단도 돌려주지 않아 생성 단계까지 가지 못한 경우.
    "no_evidence_found",
]


class ImageAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="image", max_length=160)
    mime_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
    data: str = Field(min_length=1, max_length=10_000_000)

    @field_validator("data")
    @classmethod
    def valid_base64(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("image data must be valid base64") from exc
        if len(decoded) > 5 * 1024 * 1024:
            raise ValueError("each image must be 5 MB or smaller")
        return value


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(default="", max_length=2_000)
    images: list[ImageAttachment] = Field(default_factory=list, max_length=4)
    standard_id: StandardId | None = None
    zone: Zone | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    debug: bool = False
    request_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def question_must_have_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        return value

    @model_validator(mode="after")
    def question_or_image_required(self) -> "AskRequest":
        if not self.question and not self.images:
            raise ValueError("a question or image is required")
        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    citation: str
    statement: str
    source_id: str | None = None
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    graph_path: list[Any] | None = None
    graph_hop: int | None = None
    candidate_source: str | None = None


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    reason: AnswerReason
    conclusion: str
    reasoning: list[str]
    evidence: list[EvidenceItem]
    diagnostics: dict[str, Any] | None = None


class JobCreatedResponse(BaseModel):
    job_id: str
    status: Literal["pending"] = "pending"


class JobStatusResponse(BaseModel):
    status: Literal["pending", "complete", "error"]
    result: AskResponse | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["accounting-rag-api"] = "accounting-rag-api"
