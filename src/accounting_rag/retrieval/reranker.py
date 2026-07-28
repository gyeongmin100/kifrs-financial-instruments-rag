from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class RerankConfig:
    top_k: int = 10
    batch_size: int = 12
    max_candidates: int = 48
    max_candidate_chars: int = 1800
    max_output_tokens: int = 5000
    relevance_weight: float = 0.40
    body_priority_weight: float = 0.20
    direct_evidence_weight: float = 0.30
    reference_connectivity_weight: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "top_k", "batch_size", "max_candidates", "max_candidate_chars",
            "max_output_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        weights = self.weights
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("rerank weights must be non-negative")
        if sum(weights.values()) <= 0:
            raise ValueError("at least one rerank weight must be positive")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "question_relevance": self.relevance_weight,
            "standard_body_priority": self.body_priority_weight,
            "direct_evidence": self.direct_evidence_weight,
            "reference_connectivity": self.reference_connectivity_weight,
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "RerankConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = raw.get("rerank", {})
        weight_values = values.get("weights", {})
        return cls(
            top_k=int(values.get("top_k", 10)),
            batch_size=int(values.get("batch_size", 12)),
            max_candidates=int(values.get("max_candidates", 48)),
            max_candidate_chars=int(values.get("max_candidate_chars", 1800)),
            max_output_tokens=int(values.get("max_output_tokens", 5000)),
            relevance_weight=float(weight_values.get("question_relevance", 0.40)),
            body_priority_weight=float(weight_values.get("standard_body_priority", 0.20)),
            direct_evidence_weight=float(weight_values.get("direct_evidence", 0.30)),
            reference_connectivity_weight=float(
                weight_values.get("reference_connectivity", 0.10)
            ),
        )


_SCORE_FIELDS = (
    "question_relevance",
    "standard_body_priority",
    "direct_evidence",
    "reference_connectivity",
)

_RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rankings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "question_relevance": {"type": "integer"},
                    "standard_body_priority": {"type": "integer"},
                    "direct_evidence": {"type": "integer"},
                    "reference_connectivity": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["chunk_id", *_SCORE_FIELDS, "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rankings"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You rerank evidence for Korean K-IFRS financial-instrument questions.
Score every supplied candidate independently on a 0-5 integer scale. Return each
chunk_id exactly once and never invent an id.

Rubric:
- question_relevance: how directly the candidate addresses the user's question.
- standard_body_priority: prioritize operative standard text; application guidance
  is next, while examples, implementation guidance, and basis material are support.
- direct_evidence: whether the quoted text itself supports a conclusion without
  unsupported inference. A keyword-only match is weak.
- reference_connectivity: whether explicit paragraph references or supplied graph
  paths connect this evidence to rules, conditions, exceptions, or related evidence.
  Do not reward graph popularity by itself.

Use candidate metadata only as evidence. Give a short Korean reason grounded in the
candidate. If unrelated, assign low scores instead of omitting the candidate."""


def _unique_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        chunk_id = str(candidate.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("every rerank candidate must have a non-empty chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        copied = dict(candidate)
        copied["chunk_id"] = chunk_id
        unique.append(copied)
    return unique


def _candidate_payload(candidate: Mapping[str, Any], max_chars: int) -> dict[str, Any]:
    text = str(candidate.get("contextualized_text") or candidate.get("text") or "")
    metadata_keys = (
        "standard_id", "zone", "chunk_type", "citation_label", "paragraph_id",
        "source_channels", "rrf_score", "dense_score", "sparse_score",
        "graph_distance", "graph_path", "graph_relations", "relation_types",
    )
    return {
        "chunk_id": candidate["chunk_id"],
        "text": text[:max_chars],
        "text_truncated": len(text) > max_chars,
        "metadata": {
            key: candidate[key]
            for key in metadata_keys
            if key in candidate and candidate[key] is not None
        },
    }


def _validate_rankings(payload: Any, expected_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("rankings"), list):
        raise ValueError("rerank response must contain a rankings array")
    rankings = payload["rankings"]
    ids = [item.get("chunk_id") for item in rankings if isinstance(item, dict)]
    if len(rankings) != len(expected_ids) or len(ids) != len(rankings):
        raise ValueError("rerank response omitted or malformed a candidate")
    if len(set(ids)) != len(ids):
        raise ValueError("rerank response contains duplicate chunk_id values")
    if set(ids) != set(expected_ids):
        raise ValueError("rerank response contains missing or unknown chunk_id values")
    validated: list[dict[str, Any]] = []
    for item in rankings:
        checked = {"chunk_id": str(item["chunk_id"])}
        for field in _SCORE_FIELDS:
            score = item.get(field)
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 5:
                raise ValueError(f"{field} must be an integer from 0 to 5")
            checked[field] = score
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("every rerank result must include a non-empty reason")
        checked["reason"] = reason.strip()
        validated.append(checked)
    return validated


class OpenAIReranker:
    """Score hybrid and graph-expanded candidates with schema-constrained output."""

    def __init__(self, client: Any, *, model: str,
                 config: RerankConfig | None = None) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.client = client
        self.model = model
        self.config = config or RerankConfig()

    def _score_batch(self, question: str,
                     batch: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        body = {
            "question": question,
            "candidates": [
                _candidate_payload(candidate, self.config.max_candidate_chars)
                for candidate in batch
            ],
        }
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(body, ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "kifrs_evidence_rerank",
                    "strict": True,
                    "schema": _RERANK_SCHEMA,
                }
            },
            max_output_tokens=self.config.max_output_tokens,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("rerank response has no structured output text")
        decoded = json.loads(output_text)
        return _validate_rankings(decoded, [str(item["chunk_id"]) for item in batch])

    def _weighted_score(self, scores: Mapping[str, int]) -> float:
        weights = self.config.weights
        weight_total = sum(weights.values())
        total = sum(scores[field] * weights[field] for field in _SCORE_FIELDS)
        return round(total / (5 * weight_total) * 100, 2)

    @staticmethod
    def _fallback(candidates: Sequence[Mapping[str, Any]], top_k: int,
                  reason: str) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates[:top_k], start=1):
            item = dict(candidate)
            item.update({
                "original_rank": rank,
                "rerank_rank": rank,
                "rerank_score": None,
                "rerank_scores": None,
                "rerank_reason": reason,
                "rerank_status": "fallback",
            })
            output.append(item)
        return output

    def rerank(self, question: str, candidates: Sequence[Mapping[str, Any]],
               *, top_k: int | None = None) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        unique = _unique_candidates(candidates)[:self.config.max_candidates]
        if not unique:
            return []
        limit = min(top_k or self.config.top_k, len(unique))
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")
        original_rank = {item["chunk_id"]: rank for rank, item in enumerate(unique, 1)}
        try:
            scored: dict[str, dict[str, Any]] = {}
            for start in range(0, len(unique), self.config.batch_size):
                batch = unique[start : start + self.config.batch_size]
                for result in self._score_batch(question, batch):
                    scored[result["chunk_id"]] = result
            if set(scored) != set(original_rank):
                raise ValueError("rerank batches did not cover every candidate exactly once")
        except Exception as error:
            return self._fallback(
                unique, limit, f"기존 순서 유지: {type(error).__name__}"
            )

        output: list[dict[str, Any]] = []
        for candidate in unique:
            chunk_id = candidate["chunk_id"]
            result = scored[chunk_id]
            scores = {field: result[field] for field in _SCORE_FIELDS}
            item = dict(candidate)
            item.update({
                "original_rank": original_rank[chunk_id],
                "rerank_score": self._weighted_score(scores),
                "rerank_scores": scores,
                "rerank_reason": result["reason"],
                "rerank_status": "scored",
            })
            output.append(item)
        output.sort(key=lambda item: (
            -item["rerank_score"],
            -item["rerank_scores"]["direct_evidence"],
            -item["rerank_scores"]["question_relevance"],
            item["original_rank"],
        ))
        for rank, item in enumerate(output[:limit], start=1):
            item["rerank_rank"] = rank
        return output[:limit]
