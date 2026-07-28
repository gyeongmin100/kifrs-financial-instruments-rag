from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from accounting_rag.retrieval.hybrid import normalize_standard_id


# 청킹 단계에서 한 문단의 하위항목(⑴ → ㈎ → ①)은 부모 줄기와 함께 한 청크에 담긴다.
# 다만 문단이 길면 크기 때문에 여러 청크로 갈리는데(전체 문단의 약 3%), 그때 뒷부분만
# 검색에 걸리면 앞부분이 빠진 채로 답변이 만들어진다. 같은 문단에서 나온 나머지 청크를
# 함께 붙여 그 빈틈을 메운다.
_SIBLING_QUERY = """
MATCH (hit:Chunk)-[:DERIVED_FROM]->(p:Paragraph)<-[:DERIVED_FROM]-(sibling:Chunk)
WHERE hit.chunk_id IN $chunk_ids
  AND sibling.searchable = true
  AND NOT sibling.chunk_id IN $chunk_ids
  AND ($standard_id IS NULL OR sibling.standard_id = $standard_id)
  AND ($zone IS NULL OR sibling.zone = $zone)
RETURN DISTINCT
  sibling.chunk_id AS chunk_id,
  sibling.text AS text,
  sibling.contextualized_text AS contextualized_text,
  sibling.citation_label AS citation_label,
  sibling.standard_id AS standard_id,
  sibling.zone AS zone,
  sibling.chunk_type AS chunk_type,
  sibling.pdf_page_start AS pdf_page_start,
  sibling.pdf_page_end AS pdf_page_end
LIMIT $limit
"""


@dataclass(frozen=True)
class RetrievalConfig:
    """How many chunks reach the answer generator."""

    top_k: int = 12
    max_siblings: int = 8

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.max_siblings < 0:
            raise ValueError("max_siblings must not be negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RetrievalConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        section = values.get("retrieval", {}) or {}
        defaults = cls()
        return cls(
            top_k=int(section.get("top_k", defaults.top_k)),
            max_siblings=int(section.get("max_siblings", defaults.max_siblings)),
        )


class RetrievalPipeline:
    """Hybrid search, then complete any paragraph that was split across chunks."""

    def __init__(self, hybrid_retriever: Any, driver: Any, *, database: str | None = None,
                 config: RetrievalConfig | None = None) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.driver = driver
        self.database = database
        self.config = config or RetrievalConfig()

    def _siblings(self, chunk_ids: Sequence[str], standard_id: str | None,
                  zone: str | None) -> list[dict[str, Any]]:
        if not chunk_ids or self.config.max_siblings <= 0:
            return []
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                _SIBLING_QUERY,
                chunk_ids=list(chunk_ids),
                standard_id=standard_id,
                zone=zone,
                limit=self.config.max_siblings,
            )
            return [dict(row) for row in rows]

    def retrieve(self, question: str, *, standard_id: str | None = None,
                 zone: str | None = None, top_k: int | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        normalized_standard = normalize_standard_id(standard_id)
        limit = top_k if top_k and top_k > 0 else self.config.top_k

        seeds = self.hybrid_retriever.search(
            question, standard_id=normalized_standard, zone=zone
        )[:limit]
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for seed in seeds:
            chunk_id = str(seed.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            candidate = dict(seed)
            candidate["chunk_id"] = chunk_id
            candidate["candidate_source"] = "hybrid"
            results.append(candidate)
            seen.add(chunk_id)

        siblings: list[dict[str, Any]] = []
        for row in self._siblings(list(seen), normalized_standard, zone):
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            row["chunk_id"] = chunk_id
            row["candidate_source"] = "sibling"
            siblings.append(row)
            seen.add(chunk_id)

        results.extend(siblings)
        return {
            "question": question,
            "filters": {"standard_id": normalized_standard, "zone": zone},
            "seed_count": len(seeds),
            "sibling_count": len(siblings),
            "result_count": len(results),
            "results": results,
        }
