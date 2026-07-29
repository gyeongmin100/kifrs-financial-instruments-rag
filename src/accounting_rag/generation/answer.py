from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


logger = logging.getLogger(__name__)


class AnswerGenerationError(RuntimeError):
    """The model call or its structured response could not be validated."""


@dataclass(frozen=True)
class AnswerConfig:
    # 검색이 돌려주는 최대 개수(top_k + max_siblings)보다 크게 잡는다. 이 값이 더 작으면
    # 뒤에 붙는 형제 청크가 잘려 나가 문단 복원이 무의미해진다.
    max_candidates: int = 20
    max_candidate_chars: int = 1800
    max_context_chars: int = 14000
    max_question_chars: int = 12000
    # 추론 모델은 생각하는 데 쓴 토큰도 이 한도에 포함한다. 값이 작으면 JSON이 문장
    # 중간에서 끊겨 파싱 자체가 실패한다. 한도일 뿐이라 실제 사용량만큼만 과금된다.
    max_output_tokens: int = 8000

    def __post_init__(self) -> None:
        for field in (
            "max_candidates", "max_candidate_chars", "max_context_chars",
            "max_question_chars", "max_output_tokens",
        ):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be greater than zero")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnswerConfig":
        values = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("answer", {})
        defaults = cls()
        return cls(**{
            field: int(values.get(field, getattr(defaults, field)))
            for field in (
                "max_candidates", "max_candidate_chars", "max_context_chars",
                "max_question_chars", "max_output_tokens",
            )
        })


_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string"},
        "reasoning": {"type": "array", "items": {"type": "string"}},
        # 인용한 evidence_id만 받는다. 출처와 원문은 우리가 만든 목록에서 채우므로
        # 모델에게 되묻지 않는다. 물어보지 않은 값은 틀릴 수도, 지어낼 수도 없다.
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["conclusion", "reasoning", "evidence"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """당신은 K-IFRS 제1032호, 제1039호, 제1107호, 제1109호 금융상품 질의에 답하는 회계 전문가다.
출력은 결론(conclusion), 판단과정(reasoning), 근거(evidence)의 세 필드로 작성한다.

규칙:
1. 제공된 evidence를 회계기준 판단의 최우선 근거로 사용한다.
2. 질문에 포함된 사실, 숫자, 표, 보기 등의 전제조건은 입력정보로 사용한다.
3. evidence와 충돌하지 않는 범위에서 일반 회계 지식, 계산과 논리적 추론을 사용할 수 있다. 충돌하면 evidence를 따른다.
4. evidence로 뒷받침되는 회계 판단에는 [E1] 같은 evidence_id를 붙이고, evidence에는 실제 인용한 ID만 나열한다.
5. evidence에 없는 내용을 evidence가 규정한 것처럼 인용하거나 근거의 의미를 확대하지 않는다.
6. 일부만 답할 수 있어도 확인 가능한 부분은 답하고, 부족한 정보만 구체적으로 밝힌다.
7. 보기나 계산이 있으면 주요 계산과정과 최종 선택지를 명시한다.
8. 금융상품 기준서 범위를 벗어난 질문에는 범위를 벗어났다고 설명한다.
9. 결론은 두괄식으로 작성한다. 정답, 최종 선택지 또는 핵심 회계처리를 첫 문장에 직접 제시하고, 부연 설명과 계산과정은 판단과정에서 작성한다.
10. 한국어로 간결하게 작성한다.

첨부된 원본 이미지가 제공되면 이미지의 사실, 숫자, 표와 보기를 직접 확인하여 답한다."""

_MARKER_RE = re.compile(r"\[(E\d+)\]")


def _citation(candidate: Mapping[str, Any], chunk_id: str) -> str:
    direct = str(candidate.get("citation_label") or candidate.get("citation") or "").strip()
    if direct:
        return direct
    standard = str(candidate.get("standard_id") or "").strip()
    paragraph = str(candidate.get("paragraph_id") or "").strip()
    if standard and paragraph:
        return f"K-IFRS 제{standard}호 {paragraph}"
    if standard:
        return f"K-IFRS 제{standard}호 ({chunk_id})"
    return chunk_id


def prepare_evidence_catalog(
    candidates: Sequence[Mapping[str, Any]], config: AnswerConfig,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    total_chars = 0
    for candidate in candidates[:config.max_candidates]:
        chunk_id = str(candidate.get("chunk_id") or candidate.get("node_id") or "").strip()
        if not chunk_id:
            raise ValueError("every answer candidate must have a non-empty id")
        if chunk_id in seen_chunks:
            raise ValueError(f"duplicate answer candidate id: {chunk_id}")
        seen_chunks.add(chunk_id)
        text = str(candidate.get("contextualized_text") or candidate.get("text") or "").strip()
        if not text or total_chars >= config.max_context_chars:
            continue
        allowed = min(config.max_candidate_chars, config.max_context_chars - total_chars)
        excerpt = text[:allowed]
        total_chars += len(excerpt)
        item: dict[str, Any] = {
            "evidence_id": f"E{len(prepared) + 1}",
            "source_id": chunk_id,
            "citation": _citation(candidate, chunk_id),
            "statement": excerpt,
        }
        for key in (
            "pdf_page_start", "pdf_page_end", "graph_path", "graph_hop",
            "candidate_source",
        ):
            value = candidate.get(key)
            if value is not None:
                item[key] = value
        prepared.append(item)
    return prepared


_EVIDENCE_KEYS = (
    "source_id", "pdf_page_start", "pdf_page_end", "candidate_source",
)


def _assemble_answer(
    payload: Any, offered: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """모델 응답을 화면에 낼 형태로 조립한다.

    거절하거나 폐기하지 않는다. 모델에게서 받는 것은 결론·판단과정·evidence_id뿐이고,
    출처와 원문은 우리가 만든 목록에서 되찾아 채우므로 변조될 수 없다.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("answer payload must be an object")
    conclusion = str(payload.get("conclusion") or "").strip()
    if not conclusion:
        raise ValueError("answer conclusion must not be empty")
    reasoning = [
        str(step).strip()
        for step in (payload.get("reasoning") or [])
        if str(step).strip()
    ]

    # 선언한 목록과 본문에 실제로 찍힌 [E1] 마커를 합친다. 둘 중 하나만 보면
    # 본문은 인용했는데 목록에서 빠진 근거가 카드에서 사라진다.
    cited = [str(item).strip() for item in (payload.get("evidence") or [])]
    for text in (conclusion, *reasoning):
        cited.extend(_MARKER_RE.findall(text))

    offered_by_id = {str(item["evidence_id"]): item for item in offered}
    evidence: list[dict[str, Any]] = []
    for evidence_id in dict.fromkeys(cited):
        source = offered_by_id.get(evidence_id)
        if source is None:
            continue
        item: dict[str, Any] = {
            "evidence_id": evidence_id,
            "citation": str(source["citation"]),
            "statement": str(source["statement"]),
        }
        for key in _EVIDENCE_KEYS:
            if source.get(key) is not None:
                item[key] = source[key]
        evidence.append(item)
    return {"conclusion": conclusion, "reasoning": reasoning, "evidence": evidence}


class OpenAIAnswerGenerator:
    """Generate an answer whose citations are filled in from the offered catalog."""

    def __init__(self, client: Any, *, model: str,
                 config: AnswerConfig | None = None) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.client = client
        self.model = model
        self.config = config or AnswerConfig()

    def generate(self, question: str,
                 candidates: Sequence[Mapping[str, Any]], *,
                 image_urls: Sequence[str] = ()) -> dict[str, Any]:
        question = question.strip()
        if not question and not image_urls:
            raise ValueError("question must not be empty")
        if len(question) > self.config.max_question_chars:
            raise ValueError("question exceeds max_question_chars")
        offered = prepare_evidence_catalog(candidates, self.config)
        if not offered:
            raise AnswerGenerationError("no usable evidence candidates")
        body = {"evidence": offered}
        if question:
            body["question"] = question
        user_content: str | list[dict[str, str]] = json.dumps(
            body, ensure_ascii=False,
        )
        if image_urls:
            user_content = [{"type": "input_text", "text": user_content}]
            user_content.extend({
                "type": "input_image",
                "image_url": image_url,
                "detail": "original",
            } for image_url in image_urls)
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "kifrs_grounded_answer",
                        "strict": True,
                        "schema": _ANSWER_SCHEMA,
                    }
                },
                max_output_tokens=self.config.max_output_tokens,
                store=False,
            )
            output_text = getattr(response, "output_text", None)
            if not isinstance(output_text, str) or not output_text.strip():
                raise ValueError("answer response has no structured output text")
            return _assemble_answer(json.loads(output_text), offered)
        except Exception as error:
            logger.warning("answer generation failed: %s: %s", type(error).__name__, error)
            raise AnswerGenerationError("answer generation failed") from error
