from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CitationVerificationReport:
    """Deterministic integrity checks for evidence cited by a generated answer."""

    valid: bool
    errors: tuple[str, ...]
    used_evidence_ids: tuple[str, ...]
    unused_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "used_evidence_ids": list(self.used_evidence_ids),
            "unused_evidence_ids": list(self.unused_evidence_ids),
        }


_INSUFFICIENT_STATUSES = {"insufficient", "불충분"}
_SUPPORTED_STATUSES = {"sufficient", "answerable", "충분", *_INSUFFICIENT_STATUSES}
_INSUFFICIENT_PREFIX = "근거 부족:"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return _text(
        candidate.get("evidence_id")
        or candidate.get("chunk_id")
        or candidate.get("node_id")
    )


def _candidate_citation(candidate: Mapping[str, Any]) -> str:
    return _text(candidate.get("citation_label") or candidate.get("citation"))


def _reasoning_lines(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(not isinstance(item, str) or not item.strip() for item in value):
            return []
        return [item.strip() for item in value]
    return []


def _answer_evidence(answer: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = answer.get("evidence", answer.get("citations", []))
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("answer evidence must be a sequence")
    if any(not isinstance(item, Mapping) for item in value):
        raise TypeError("answer evidence entries must be objects")
    return list(value)


def _retrieval_candidates(
    candidates: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if isinstance(candidates, Mapping):
        value = candidates.get("results", candidates.get("candidates", []))
    else:
        value = candidates
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("retrieval candidates must be a sequence")
    return [item for item in value if isinstance(item, Mapping)]


def verify_citations(
    answer: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> CitationVerificationReport:
    """Verify that an answer only cites intact evidence from retrieval results.

    Candidate identity is resolved in ``evidence_id``, ``chunk_id``, ``node_id``
    order. Candidate citation text is resolved from ``citation_label`` and then
    ``citation`` so both hybrid chunks and graph evidence are supported.
    """

    errors: list[str] = []
    candidate_rows = _retrieval_candidates(candidates)
    candidate_by_id: dict[str, Mapping[str, Any]] = {}
    candidate_order: list[str] = []
    duplicate_candidate_ids: set[str] = set()

    for candidate in candidate_rows:
        evidence_id = _candidate_id(candidate)
        if not evidence_id:
            errors.append("retrieval candidate has no evidence_id")
            continue
        if evidence_id in candidate_by_id:
            duplicate_candidate_ids.add(evidence_id)
            continue
        candidate_by_id[evidence_id] = candidate
        candidate_order.append(evidence_id)

    for evidence_id in sorted(duplicate_candidate_ids):
        errors.append(f"duplicate retrieval evidence_id: {evidence_id}")

    reasoning_lines = _reasoning_lines(answer.get("reasoning"))
    conclusion_value = answer.get("conclusion")
    conclusion = conclusion_value.strip() if isinstance(conclusion_value, str) else ""
    explicit_status = _text(answer.get("status")).lower()
    inferred_insufficient = (
        not explicit_status
        and conclusion.startswith(_INSUFFICIENT_PREFIX)
        and bool(reasoning_lines)
        and all(line.startswith(_INSUFFICIENT_PREFIX) for line in reasoning_lines)
    )
    status = explicit_status or ("insufficient" if inferred_insufficient else "sufficient")
    if status not in _SUPPORTED_STATUSES:
        errors.append(f"unsupported answer status: {status or '<empty>'}")

    if not conclusion:
        errors.append("answer conclusion is empty")
    if not reasoning_lines:
        errors.append("answer reasoning is empty")

    try:
        answer_evidence = _answer_evidence(answer)
    except TypeError as exc:
        errors.append(str(exc))
        answer_evidence = []

    if status not in _INSUFFICIENT_STATUSES and not answer_evidence:
        errors.append("sufficient answer has no evidence")

    used_ids: list[str] = []
    seen_used: set[str] = set()
    duplicate_used: set[str] = set()

    for index, item in enumerate(answer_evidence):
        evidence_id = _text(item.get("evidence_id") or item.get("chunk_id"))
        if not evidence_id:
            errors.append(f"answer evidence[{index}] has no evidence_id")
            continue
        if evidence_id in seen_used:
            duplicate_used.add(evidence_id)
            continue
        seen_used.add(evidence_id)
        used_ids.append(evidence_id)

        candidate = candidate_by_id.get(evidence_id)
        if candidate is None:
            errors.append(f"unknown answer evidence_id: {evidence_id}")
        else:
            citation_value = item.get("citation") or item.get("citation_label")
            cited = citation_value.strip() if isinstance(citation_value, str) else ""
            expected = _candidate_citation(candidate)
            if not expected:
                errors.append(f"retrieval evidence has no citation: {evidence_id}")
            elif cited != expected:
                errors.append(
                    f"citation mismatch for {evidence_id}: "
                    f"expected {expected!r}, got {cited!r}"
                )

        statement = item.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            errors.append(f"evidence statement is empty: {evidence_id}")

    for evidence_id in sorted(duplicate_used):
        errors.append(f"duplicate answer evidence_id: {evidence_id}")

    unused_ids = [item for item in candidate_order if item not in seen_used]
    return CitationVerificationReport(
        valid=not errors,
        errors=tuple(errors),
        used_evidence_ids=tuple(used_ids),
        unused_evidence_ids=tuple(unused_ids),
    )
