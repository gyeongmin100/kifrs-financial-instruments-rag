from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import yaml


DEFAULT_STANDARD_IDS = ("1032", "1039", "1107", "1109")
_PID = r"(?:[A-Z]{0,5}\.?)?\d+[A-Z]?(?:\.\d+[A-Z]?)*"
_PVAL_RE = re.compile(rf"^(?P<start>{_PID})(?:\s*[~∼～\-–—]\s*(?P<end>{_PID}))?$", re.I)
_PGROUP_RE = re.compile(
    rf"(?:문단|paragraphs?|paras?\.?)[ \t]*(?P<first>{_PID})"
    rf"(?P<tail>(?:[ \t]*(?:[~∼～\-–—]|,|，|및|와|과|또는)[ \t]*(?:문단[ \t]*)?{_PID})*)", re.I,
)
_JE_P_RE = re.compile(rf"제\s*(?P<value>{_PID})\s*항", re.I)
_PID_RE = re.compile(_PID, re.I)
_STANDARD_PATTERNS = (
    (re.compile(r"(?:K[\s-]*IFRS|기업회계기준서)?\s*제?\s*(?P<n>\d{4})\s*호", re.I), None),
    (re.compile(r"K[\s-]*IFRS\s*(?P<n>\d{4})", re.I), None),
    (re.compile(r"IFRS\s*(?P<n>7|9)(?!\d)", re.I), {"7": "1107", "9": "1109"}),
    (re.compile(r"IAS\s*(?P<n>32|39)(?!\d)", re.I), {"32": "1032", "39": "1039"}),
)
@dataclass(frozen=True)
class QuestionAnalysis:
    requested_standard_ids: tuple[str, ...]
    requested_paragraphs: tuple[str, ...]
    concepts: tuple[str, ...]
    subquestions: tuple[str, ...]
    search_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuestionAnalysisConfig:
    allowed_standard_ids: tuple[str, ...] = DEFAULT_STANDARD_IDS
    max_output_tokens: int = 1800
    max_question_chars: int = 12000

    def __post_init__(self) -> None:
        if not self.allowed_standard_ids or any(not re.fullmatch(r"\d{4}", x) for x in self.allowed_standard_ids):
            raise ValueError("allowed_standard_ids must contain four-digit strings")
        if len(set(self.allowed_standard_ids)) != len(self.allowed_standard_ids):
            raise ValueError("allowed_standard_ids must not contain duplicates")
        if self.max_output_tokens <= 0 or self.max_question_chars <= 0:
            raise ValueError("query-analysis size limits must be positive")

    @classmethod
    def from_yaml(cls, path: Path) -> "QuestionAnalysisConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = raw.get("query_analysis", {})
        return cls(
            allowed_standard_ids=tuple(str(x) for x in value.get("allowed_standard_ids", DEFAULT_STANDARD_IDS)),
            max_output_tokens=int(value.get("max_output_tokens", 1800)),
            max_question_chars=int(value.get("max_question_chars", 12000)),
        )


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def extract_explicit_standards(question: str) -> tuple[str, ...]:
    found: list[tuple[int, str]] = []
    text = unicodedata.normalize("NFKC", question)
    for pattern, aliases in _STANDARD_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group("n")
            found.append((match.start(), aliases[raw] if aliases else raw))
    return _unique([x for _, x in sorted(found)])


def _normalize_paragraph(value: str) -> str:
    match = _PVAL_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"invalid paragraph identifier: {value!r}")
    start, end = match.group("start").upper(), match.group("end")
    return f"{start}~{end.upper()}" if end else start


def extract_explicit_paragraphs(question: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", question)
    found: list[tuple[int, str]] = []
    for group in _PGROUP_RE.finditer(text):
        group_text = group.group("first") + group.group("tail")
        ids = list(_PID_RE.finditer(group_text))
        index = 0
        while index < len(ids):
            current, value = ids[index], ids[index].group(0)
            if index + 1 < len(ids) and re.search(r"[~∼～\-–—]", group_text[current.end():ids[index + 1].start()]):
                value += "~" + ids[index + 1].group(0)
                index += 1
            found.append((group.start() + current.start(), _normalize_paragraph(value)))
            index += 1
    found.extend((m.start(), _normalize_paragraph(m.group("value"))) for m in _JE_P_RE.finditer(text))
    return _unique([x for _, x in sorted(found)])


def _schema(allowed: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "requested_standard_ids": {"type": "array", "items": {"type": "string", "enum": list(allowed)}},
            "requested_paragraphs": {"type": "array", "items": {"type": "string", "pattern": _PVAL_RE.pattern}},
            "concepts": {"type": "array", "items": {"type": "string"}},
            "subquestions": {"type": "array", "items": {"type": "string"}},
            "search_query": {"type": "string"},
        },
        "required": ["requested_standard_ids", "requested_paragraphs", "concepts", "subquestions", "search_query"],
        "additionalProperties": False,
    }


_PROMPT = """Analyze a Korean K-IFRS financial-instrument question only for retrieval; do not answer it.
Copy deterministic standard IDs and paragraph IDs exactly. Never invent them.
Do not classify the response format. Preserve any choices or calculations inside search_query when relevant.
Provide concise concepts, subquestions, and one self-contained search_query."""


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"{field} must be a string array")
    result = tuple(x.strip() for x in value)
    if any(not x for x in result) or len(set(result)) != len(result):
        raise ValueError(f"{field} contains empty or duplicate values")
    return result


def _validate(payload: Any, standards: tuple[str, ...], paragraphs: tuple[str, ...],
              allowed: set[str]) -> QuestionAnalysis:
    required = {"requested_standard_ids", "requested_paragraphs", "concepts", "subquestions", "search_query"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("question-analysis response has missing or unknown fields")
    got_standards = _strings(payload["requested_standard_ids"], "requested_standard_ids")
    if any(x not in allowed for x in got_standards):
        raise ValueError("unknown standard")
    got_paragraphs = tuple(_normalize_paragraph(x) for x in _strings(payload["requested_paragraphs"], "requested_paragraphs"))
    if len(set(got_paragraphs)) != len(got_paragraphs):
        raise ValueError("duplicate paragraph")
    if got_standards != standards or got_paragraphs != paragraphs:
        raise ValueError("LLM changed explicit constraints")
    search_query = payload["search_query"]
    if not isinstance(search_query, str) or not search_query.strip():
        raise ValueError("invalid search query")
    return QuestionAnalysis(got_standards, got_paragraphs,
                            _strings(payload["concepts"], "concepts"),
                            _strings(payload["subquestions"], "subquestions"),
                            search_query.strip())


def _fallback(question: str, standards: tuple[str, ...],
              paragraphs: tuple[str, ...]) -> QuestionAnalysis:
    return QuestionAnalysis(standards, paragraphs, (), (), question)


class OpenAIQuestionAnalyzer:
    def __init__(self, client: Any, *, model: str, config: QuestionAnalysisConfig | None = None) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.client, self.model, self.config = client, model, config or QuestionAnalysisConfig()

    def analyze(self, question: str) -> QuestionAnalysis:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if len(question) > self.config.max_question_chars:
            raise ValueError("question exceeds max_question_chars")
        standards = extract_explicit_standards(question)
        paragraphs = extract_explicit_paragraphs(question)
        unknown = set(standards) - set(self.config.allowed_standard_ids)
        if unknown:
            raise ValueError(f"question names unsupported standards: {sorted(unknown)}")
        deterministic = {"explicit_standard_ids": standards, "explicit_paragraphs": paragraphs}
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": "system", "content": _PROMPT}, {"role": "user", "content": json.dumps({"question": question, "deterministic": deterministic}, ensure_ascii=False)}],
                text={"format": {"type": "json_schema", "name": "kifrs_question_analysis", "strict": True, "schema": _schema(self.config.allowed_standard_ids)}},
                max_output_tokens=self.config.max_output_tokens, store=False,
            )
            output = getattr(response, "output_text", None)
            if not isinstance(output, str) or not output.strip():
                raise ValueError("response has no structured output")
            return _validate(json.loads(output), standards, paragraphs,
                             set(self.config.allowed_standard_ids))
        except Exception:
            return _fallback(question, standards, paragraphs)
