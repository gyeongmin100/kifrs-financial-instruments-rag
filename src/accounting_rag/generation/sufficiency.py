from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import yaml


@dataclass(frozen=True)
class SufficiencyConfig:
    min_candidates: int = 3
    min_distinct_evidence_ids: int = 2
    min_non_empty_text: int = 2
    min_cited_candidates: int = 2
    min_scored_candidates: int = 2
    min_qualified_evidence: int = 2
    min_question_relevance: int = 3
    min_direct_evidence: int = 3
    semantic_confidence_threshold: float = 0.65
    max_evidence_chars: int = 1800
    max_output_tokens: int = 2500

    def __post_init__(self) -> None:
        count_fields = (
            "min_candidates",
            "min_distinct_evidence_ids",
            "min_non_empty_text",
            "min_cited_candidates",
            "min_scored_candidates",
            "min_qualified_evidence",
            "max_evidence_chars",
            "max_output_tokens",
        )
        if any(getattr(self, name) <= 0 for name in count_fields):
            raise ValueError("sufficiency count and size thresholds must be positive")
        for name in ("min_question_relevance", "min_direct_evidence"):
            value = getattr(self, name)
            if not 0 <= value <= 5:
                raise ValueError(f"{name} must be between 0 and 5")
        if not 0 <= self.semantic_confidence_threshold <= 1:
            raise ValueError("semantic_confidence_threshold must be between 0 and 1")

    @classmethod
    def from_yaml(cls, path: Path) -> "SufficiencyConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = raw.get("sufficiency", {})
        return cls(
            min_candidates=int(values.get("min_candidates", 3)),
            min_distinct_evidence_ids=int(values.get("min_distinct_evidence_ids", 2)),
            min_non_empty_text=int(values.get("min_non_empty_text", 2)),
            min_cited_candidates=int(values.get("min_cited_candidates", 2)),
            min_scored_candidates=int(values.get("min_scored_candidates", 2)),
            min_qualified_evidence=int(values.get("min_qualified_evidence", 2)),
            min_question_relevance=int(values.get("min_question_relevance", 3)),
            min_direct_evidence=int(values.get("min_direct_evidence", 3)),
            semantic_confidence_threshold=float(
                values.get("semantic_confidence_threshold", 0.65)
            ),
            max_evidence_chars=int(values.get("max_evidence_chars", 1800)),
            max_output_tokens=int(values.get("max_output_tokens", 2500)),
        )


class SemanticJudge(Protocol):
    def judge(
        self, question: str, evidence: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]: ...


_SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "covered": {"type": "boolean"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "covered", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "supported_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "missing_aspects": {"type": "array", "items": {"type": "string"}},
        "sufficient": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "claim_coverage",
        "supported_evidence_ids",
        "missing_aspects",
        "sufficient",
        "confidence",
    ],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """You determine whether retrieved K-IFRS evidence is sufficient to answer a
Korean financial-instrument question. Break the question into its core claims and
conditions. Mark a claim covered only when the supplied evidence directly supports it.
Use only the supplied evidence IDs, never invent an ID, and report every material gap.
Return sufficient=true only when all material claims, conditions, and exceptions needed
for a grounded answer are covered. Do not rely on outside knowledge."""


def _text(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("contextualized_text") or candidate.get("text") or "").strip()


def _has_citation(candidate: Mapping[str, Any]) -> bool:
    return any(
        str(candidate.get(key) or "").strip()
        for key in ("citation_label", "citation", "graph_node_id")
    )


def _scores(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    value = candidate.get("rerank_scores")
    return value if isinstance(value, Mapping) else {}


def _is_scored(candidate: Mapping[str, Any]) -> bool:
    scores = _scores(candidate)
    return (
        candidate.get("rerank_status") == "scored"
        and isinstance(scores.get("question_relevance"), (int, float))
        and not isinstance(scores.get("question_relevance"), bool)
        and isinstance(scores.get("direct_evidence"), (int, float))
        and not isinstance(scores.get("direct_evidence"), bool)
    )


def _validate_semantic_payload(
    payload: Any, allowed_ids: set[str]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("semantic judge response must be an object")
    required = {
        "claim_coverage", "supported_evidence_ids", "missing_aspects",
        "sufficient", "confidence",
    }
    if not required.issubset(payload):
        raise ValueError("semantic judge response is missing required fields")
    coverage = payload["claim_coverage"]
    supported = payload["supported_evidence_ids"]
    missing = payload["missing_aspects"]
    if not isinstance(coverage, list) or not isinstance(supported, list) or not isinstance(missing, list):
        raise ValueError("semantic judge array fields must be arrays")
    if not isinstance(payload["sufficient"], bool):
        raise ValueError("semantic sufficient must be boolean")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("semantic confidence must be between 0 and 1")
    if any(not isinstance(item, str) for item in supported + missing):
        raise ValueError("semantic evidence IDs and missing aspects must be strings")

    cited_ids = set(supported)
    checked_coverage: list[dict[str, Any]] = []
    for item in coverage:
        if not isinstance(item, Mapping):
            raise ValueError("claim coverage entries must be objects")
        claim = item.get("claim")
        covered = item.get("covered")
        evidence_ids = item.get("evidence_ids")
        if not isinstance(claim, str) or not claim.strip() or not isinstance(covered, bool):
            raise ValueError("claim coverage contains an invalid claim or covered flag")
        if not isinstance(evidence_ids, list) or any(
            not isinstance(evidence_id, str) for evidence_id in evidence_ids
        ):
            raise ValueError("claim evidence_ids must be a string array")
        cited_ids.update(evidence_ids)
        checked_coverage.append({
            "claim": claim.strip(),
            "covered": covered,
            "evidence_ids": list(evidence_ids),
        })
    unknown = cited_ids - allowed_ids
    if unknown:
        raise ValueError(f"semantic judge invented unknown evidence IDs: {sorted(unknown)}")
    return {
        "claim_coverage": checked_coverage,
        "supported_evidence_ids": list(supported),
        "missing_aspects": list(missing),
        "sufficient": payload["sufficient"],
        "confidence": float(confidence),
    }


class OpenAISemanticJudge:
    """Use the Responses API with strict JSON Schema output."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_evidence_chars: int = 1800,
        max_output_tokens: int = 2500,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.client = client
        self.model = model
        self.max_evidence_chars = max_evidence_chars
        self.max_output_tokens = max_output_tokens

    def judge(
        self, question: str, evidence: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        items = [
            {
                "evidence_id": str(item["chunk_id"]),
                "citation": item.get("citation_label") or item.get("citation")
                or item.get("graph_node_id"),
                "text": _text(item)[: self.max_evidence_chars],
                "rerank_scores": dict(_scores(item)),
            }
            for item in evidence
        ]
        allowed_ids = {item["evidence_id"] for item in items}
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"question": question, "evidence": items}, ensure_ascii=False
                )},
            ],
            text={"format": {
                "type": "json_schema",
                "name": "kifrs_evidence_sufficiency",
                "strict": True,
                "schema": _SEMANTIC_SCHEMA,
            }},
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ValueError("semantic judge response has no structured output text")
        return _validate_semantic_payload(json.loads(output_text), allowed_ids)


class EvidenceSufficiencyChecker:
    """Gate answer generation with deterministic checks and a semantic judge."""

    def __init__(
        self,
        semantic_judge: SemanticJudge | None,
        *,
        config: SufficiencyConfig | None = None,
    ) -> None:
        self.semantic_judge = semantic_judge
        self.config = config or SufficiencyConfig()

    def _deterministic(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        config = self.config
        ids = [str(item.get("chunk_id") or "").strip() for item in candidates]
        non_empty = [item for item in candidates if _text(item)]
        cited = [item for item in candidates if _has_citation(item)]
        scored = [item for item in candidates if _is_scored(item)]
        qualified = [
            item for item in candidates
            if str(item.get("chunk_id") or "").strip()
            and _text(item)
            and _has_citation(item)
            and _is_scored(item)
            and _scores(item)["question_relevance"] >= config.min_question_relevance
            and _scores(item)["direct_evidence"] >= config.min_direct_evidence
        ]
        checks = {
            "candidate_count": len(candidates) >= config.min_candidates,
            "distinct_evidence_ids": len({item for item in ids if item})
            >= config.min_distinct_evidence_ids,
            "non_empty_text": len(non_empty) >= config.min_non_empty_text,
            "citations": len(cited) >= config.min_cited_candidates,
            "scored_candidates": len(scored) >= config.min_scored_candidates,
            "qualified_evidence": len(qualified) >= config.min_qualified_evidence,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "counts": {
                "candidates": len(candidates),
                "distinct_evidence_ids": len({item for item in ids if item}),
                "non_empty_text": len(non_empty),
                "cited": len(cited),
                "scored": len(scored),
                "qualified": len(qualified),
            },
            "qualified_evidence_ids": [str(item["chunk_id"]) for item in qualified],
            "qualified_evidence": qualified,
        }

    def check(
        self, question: str, candidates: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        deterministic = self._deterministic(candidates)
        public_deterministic = {
            key: value for key, value in deterministic.items() if key != "qualified_evidence"
        }
        if not deterministic["passed"]:
            return {
                "sufficient": False,
                "status": "deterministic_failed",
                "confidence": 0.0,
                "deterministic": public_deterministic,
                "semantic": None,
            }
        if self.semantic_judge is None:
            return {
                "sufficient": False,
                "status": "semantic_not_run",
                "confidence": 0.0,
                "deterministic": public_deterministic,
                "semantic": None,
            }
        try:
            raw_semantic = self.semantic_judge.judge(
                question, deterministic["qualified_evidence"]
            )
            semantic = _validate_semantic_payload(
                raw_semantic, set(deterministic["qualified_evidence_ids"])
            )
        except Exception as error:
            return {
                "sufficient": False,
                "status": "semantic_error",
                "confidence": 0.0,
                "deterministic": public_deterministic,
                "semantic": {
                    "error_type": type(error).__name__,
                    "message": "semantic sufficiency validation failed",
                },
            }
        sufficient = (
            semantic["sufficient"]
            and semantic["confidence"] >= self.config.semantic_confidence_threshold
            and bool(semantic["claim_coverage"])
            and all(item["covered"] for item in semantic["claim_coverage"])
            and not semantic["missing_aspects"]
        )
        return {
            "sufficient": sufficient,
            "status": "sufficient" if sufficient else "semantic_insufficient",
            "confidence": semantic["confidence"],
            "deterministic": public_deterministic,
            "semantic": semantic,
        }
