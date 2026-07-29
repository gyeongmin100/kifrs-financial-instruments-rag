from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from accounting_rag.retrieval.hybrid import normalize_standard_id


# 청킹 단계에서 한 문단의 하위항목(⑴ → ㈎ → ①)은 부모 줄기와 함께 한 청크에 담긴다.
# 다만 문단이 길면 크기 때문에 여러 청크로 갈리는데(전체 문단의 약 3%), 그때 뒷부분만
# 검색에 걸리면 앞부분이 빠진 채로 답변이 만들어진다. 같은 문단에서 나온 나머지 청크를
# part_index 순으로 해당 seed 바로 뒤에 붙여 그 빈틈을 메운다.
_SIBLING_QUERY = """
UNWIND range(0, size($chunk_ids) - 1) AS seed_rank
WITH seed_rank, $chunk_ids[seed_rank] AS seed_chunk_id
MATCH (hit:Chunk {chunk_id: seed_chunk_id})-[:DERIVED_FROM]->(p:Paragraph)<-[:DERIVED_FROM]-(sibling:Chunk)
WHERE sibling.searchable = true
  AND NOT sibling.chunk_id IN $chunk_ids
  AND ($standard_id IS NULL OR sibling.standard_id = $standard_id)
  AND ($zone IS NULL OR sibling.zone = $zone)
WITH sibling, min(seed_rank) AS seed_rank
RETURN DISTINCT
  seed_rank,
  sibling.chunk_id AS chunk_id,
  sibling.part_index AS part_index,
  sibling.text AS text,
  sibling.contextualized_text AS contextualized_text,
  sibling.citation_label AS citation_label,
  sibling.standard_id AS standard_id,
  sibling.zone AS zone,
  sibling.chunk_type AS chunk_type,
  sibling.pdf_page_start AS pdf_page_start,
  sibling.pdf_page_end AS pdf_page_end
ORDER BY seed_rank, part_index, chunk_id
LIMIT $limit
"""


@dataclass(frozen=True)
class RetrievalConfig:
    """How many chunks reach the answer generator."""

    top_k: int = 6
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
                 zone: str | None = None, top_k: int | None = None,
                 keyword_query: str | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        normalized_standard = normalize_standard_id(standard_id)
        limit = top_k if top_k and top_k > 0 else self.config.top_k

        raw_candidate_count: int | None = None
        search_options = {"standard_id": normalized_standard, "zone": zone}
        if keyword_query is not None:
            search_options["sparse_query"] = keyword_query
        if hasattr(self.hybrid_retriever, "search_with_scores"):
            snapshot = self.hybrid_retriever.search_with_scores(
                question, **search_options,
            )
            raw_candidate_count = len({
                str(row.get("chunk_id") or "")
                for row in (*snapshot["dense"], *snapshot["sparse"])
                if row.get("chunk_id")
            })
            seeds = list(snapshot["results"])[:limit]
        else:
            seeds = self.hybrid_retriever.search(
                question, **search_options,
            )[:limit]
        seed_candidates: list[dict[str, Any]] = []
        seed_ids: set[str] = set()
        for seed in seeds:
            chunk_id = str(seed.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seed_ids:
                continue
            candidate = dict(seed)
            candidate["chunk_id"] = chunk_id
            candidate["candidate_source"] = "hybrid"
            seed_candidates.append(candidate)
            seed_ids.add(chunk_id)

        siblings_by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in self._siblings(
            [candidate["chunk_id"] for candidate in seed_candidates],
            normalized_standard,
            zone,
        ):
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seed_ids:
                continue
            row["chunk_id"] = chunk_id
            row["candidate_source"] = "sibling"
            seed_rank = int(row.pop("seed_rank", 0))
            siblings_by_seed.setdefault(seed_rank, []).append(row)

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        sibling_count = 0
        for seed_rank, seed in enumerate(seed_candidates):
            chunk_id = seed["chunk_id"]
            if chunk_id not in seen:
                results.append(seed)
                seen.add(chunk_id)
            for sibling in siblings_by_seed.get(seed_rank, []):
                sibling_id = sibling["chunk_id"]
                if sibling_id in seen:
                    continue
                results.append(sibling)
                seen.add(sibling_id)
                sibling_count += 1

        return {
            "question": question,
            "filters": {"standard_id": normalized_standard, "zone": zone},
            "seed_count": len(seeds),
            "sibling_count": sibling_count,
            "result_count": len(results),
            "raw_candidate_count": raw_candidate_count,
            "threshold_filtered_all": bool(raw_candidate_count and not seeds),
            "results": results,
        }
