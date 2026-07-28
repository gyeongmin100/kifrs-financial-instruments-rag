from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerConfig:
    # 검색이 돌려주는 최대 개수(top_k + max_siblings)보다 크게 잡는다. 이 값이 더 작으면
    # 뒤에 붙는 형제 청크가 잘려 나가 문단 복원이 무의미해진다.
    max_candidates: int = 20
    max_candidate_chars: int = 1800
    max_context_chars: int = 14000
    max_question_chars: int = 2000
    max_output_tokens: int = 3000

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
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "citation": {"type": "string"},
                    "statement": {"type": "string"},
                },
                "required": ["evidence_id", "citation", "statement"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["conclusion", "reasoning", "evidence"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """당신은 K-IFRS 금융상품 질의에 답하는 근거 중심 회계 전문가다.
반드시 제공된 evidence만 사용하고 외부 지식이나 제공되지 않은 사실을 추가하지 않는다.
출력은 결론(conclusion), 판단과정(reasoning), 근거(evidence)의 세 필드뿐이다.

규칙:
1. 결론과 판단과정의 사실 또는 회계 판단 문장 끝에는 반드시 [E1] 같은 evidence_id를 붙인다.
2. reasoning의 각 원소는 한 가지 판단 단계만 담는다.
3. evidence에는 실제로 답변에서 인용한 항목만 넣고, 제공된 evidence_id와 citation을 그대로 쓴다.
4. 근거 원문의 의미를 확대하거나 서로 다른 조건을 임의로 합치지 않는다.
5. 결론에 필요한 조건, 예외 또는 사실관계가 부족하면 단정하지 않는다. 그 문장은
   '근거 부족:'으로 시작하여 무엇이 부족한지 구체적으로 밝힌다. 이 메타 문장에는 인용이 없어도 된다.
6. 질문에 답할 근거가 전혀 없으면 결론에서 답변할 수 없다고 밝히고 evidence는 빈 배열로 둔다.
7. 질문 형식에 맞춰 스스로 답한다. 보기가 있으면 각 보기를 근거와 대조한 뒤 결론에 선택지를 명시하고,
   계산이 필요하면 판단과정에 사용한 값·식·단위와 계산 결과를 명시한다.
8. 한국어로 간결하게 작성한다."""

_MARKER_RE = re.compile(r"\[(E\d+)\]")
_INSUFFICIENT_PREFIXES = ("근거 부족:", "답변 불가:")


def _fallback() -> dict[str, Any]:
    return {
        "conclusion": "근거 부족: 검증된 답변을 생성하지 못했습니다.",
        "reasoning": ["근거 부족: 생성 결과를 검증하지 못해 추가 확인이 필요합니다."],
        "evidence": [],
    }


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


def _validate_narrative(text: str, allowed_ids: set[str]) -> None:
    if not text.strip():
        raise ValueError("answer narrative must not be empty")
    markers = _MARKER_RE.findall(text)
    unknown = set(markers) - allowed_ids
    if unknown:
        raise ValueError(f"answer contains unknown evidence ids: {sorted(unknown)}")
    if not markers and not text.strip().startswith(_INSUFFICIENT_PREFIXES):
        raise ValueError("every grounded conclusion and reasoning step must cite an evidence_id")


def _validate_answer(payload: Any, offered: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"conclusion", "reasoning", "evidence"}:
        raise ValueError("answer must contain exactly conclusion, reasoning, and evidence")
    conclusion = payload["conclusion"]
    reasoning = payload["reasoning"]
    evidence = payload["evidence"]
    if not isinstance(conclusion, str) or not isinstance(reasoning, list) or not isinstance(evidence, list):
        raise ValueError("answer fields have invalid types")
    if any(not isinstance(step, str) or not step.strip() for step in reasoning):
        raise ValueError("reasoning must contain non-empty strings")

    offered_by_id = {item["evidence_id"]: item for item in offered}
    validated_evidence: list[dict[str, str]] = []
    ids: list[str] = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"evidence_id", "citation", "statement"}:
            raise ValueError("every evidence item must have exactly the required fields")
        if not all(isinstance(item[key], str) and item[key].strip() for key in item):
            raise ValueError("evidence fields must be non-empty strings")
        evidence_id = item["evidence_id"].strip()
        if evidence_id not in offered_by_id:
            raise ValueError(f"unknown evidence_id: {evidence_id}")
        if item["citation"].strip() != offered_by_id[evidence_id]["citation"]:
            raise ValueError(f"citation does not match offered evidence: {evidence_id}")
        ids.append(evidence_id)
        validated_evidence.append({
            "evidence_id": evidence_id,
            "citation": item["citation"].strip(),
            "statement": item["statement"].strip(),
        })
    if len(ids) != len(set(ids)):
        raise ValueError("answer contains duplicate evidence_id values")

    allowed_ids = set(ids)
    _validate_narrative(conclusion, allowed_ids)
    for step in reasoning:
        _validate_narrative(step, allowed_ids)
    referenced = set(_MARKER_RE.findall(conclusion))
    for step in reasoning:
        referenced.update(_MARKER_RE.findall(step))
    if referenced != allowed_ids:
        raise ValueError("every returned evidence item must be cited in the answer")
    return {
        "conclusion": conclusion.strip(),
        "reasoning": [step.strip() for step in reasoning],
        "evidence": validated_evidence,
    }


class OpenAIAnswerGenerator:
    """Generate a citation-checked answer from reranked retrieval candidates."""

    def __init__(self, client: Any, *, model: str,
                 config: AnswerConfig | None = None) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.client = client
        self.model = model
        self.config = config or AnswerConfig()

    def generate(self, question: str,
                 candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if len(question) > self.config.max_question_chars:
            raise ValueError("question exceeds max_question_chars")
        offered = prepare_evidence_catalog(candidates, self.config)
        if not offered:
            return _fallback()
        body = {"question": question, "evidence": offered}
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
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
            return _validate_answer(json.loads(output_text), offered)
        except Exception as error:
            logger.warning("answer generation fallback: %s: %s", type(error).__name__, error)
            return _fallback()
