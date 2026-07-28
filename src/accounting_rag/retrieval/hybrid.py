from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class HybridConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    seed_top_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    embedding_dimensions: int = 3072

    def __post_init__(self) -> None:
        for name in (
            "dense_top_k", "sparse_top_k", "seed_top_k", "rrf_k", "embedding_dimensions"
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("RRF weights must be non-negative")
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("At least one RRF weight must be positive")

    @classmethod
    def from_yaml(cls, path: Path) -> "HybridConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = raw.get("hybrid", raw)
        return cls(
            dense_top_k=int(values.get("dense_top_k", 20)),
            sparse_top_k=int(values.get("sparse_top_k", 20)),
            seed_top_k=int(values.get("seed_top_k", 10)),
            rrf_k=int(values.get("rrf_k", 60)),
            dense_weight=float(values.get("dense_weight", 1.0)),
            sparse_weight=float(values.get("sparse_weight", 1.0)),
            embedding_dimensions=int(values.get("embedding_dimensions", 3072)),
        )


_LUCENE_SPECIAL = re.compile(r"(\&\&|\|\||[+\-!(){}\[\]^\"~*?:\\/])")
_STANDARD_ID = re.compile(r"(?:K[\s-]*IFRS\s*)?(?:제\s*)?(\d{4})(?:\s*호)?", re.IGNORECASE)


def escape_lucene_query(text: str) -> str:
    """Escape Lucene query-parser operators while preserving Korean text and spaces."""
    return _LUCENE_SPECIAL.sub(lambda match: "\\" + match.group(0), text)


def normalize_standard_id(value: str | None) -> str | None:
    """Accept user-facing forms such as 1109, KIFRS1109, and 제1109호."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    match = _STANDARD_ID.fullmatch(value)
    if not match:
        raise ValueError(f"Unsupported standard_id format: {value}")
    return match.group(1)


def reciprocal_rank_fusion(
    dense_rows: Sequence[Mapping[str, Any]],
    sparse_rows: Sequence[Mapping[str, Any]],
    *,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse channels while preserving their raw ranks and scores."""
    if rrf_k <= 0 or top_k <= 0:
        raise ValueError("rrf_k and top_k must be greater than zero")
    combined: dict[str, dict[str, Any]] = {}
    for channel, rows, weight in (
        ("dense", dense_rows, dense_weight),
        ("sparse", sparse_rows, sparse_weight),
    ):
        seen: set[str] = set()
        for rank, source in enumerate(rows, start=1):
            chunk_id = str(source["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            item = combined.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "standard_id": source.get("standard_id"),
                    "zone": source.get("zone"),
                    "chunk_type": source.get("chunk_type"),
                    "text": source.get("text"),
                    "contextualized_text": source.get("contextualized_text"),
                    "citation_label": source.get("citation_label"),
                    "pdf_page_start": source.get("pdf_page_start"),
                    "pdf_page_end": source.get("pdf_page_end"),
                    "search_priority": source.get("search_priority"),
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "rrf_score": 0.0,
                    "source_channels": [],
                },
            )
            if item[f"{channel}_rank"] is not None:
                continue
            item[f"{channel}_rank"] = rank
            item[f"{channel}_score"] = float(source["score"])
            item["rrf_score"] += weight / (rrf_k + rank)
            item["source_channels"].append(channel)
    for item in combined.values():
        item["source_channels"].sort()
    return sorted(
        combined.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"])
    )[:top_k]


class HybridRetriever:
    """Neo4j Dense + Sparse retrieval followed by weighted RRF seed selection."""

    _DENSE_QUERY = """
MATCH (node:Chunk)
  SEARCH node IN (
    VECTOR INDEX chunk_embedding_vector
    FOR $embedding
    LIMIT $top_k
  ) SCORE AS score
WHERE node:Chunk
  AND coalesce(node.searchable, false) = true
  AND coalesce(node.inactive, false) = false
  AND ($standard_id IS NULL OR node.standard_id = $standard_id)
  AND ($zone IS NULL OR node.zone = $zone)
RETURN node.chunk_id AS chunk_id, node.standard_id AS standard_id,
       node.zone AS zone, node.chunk_type AS chunk_type, node.text AS text,
       node.contextualized_text AS contextualized_text,
       node.citation_label AS citation_label,
       node.pdf_page_start AS pdf_page_start, node.pdf_page_end AS pdf_page_end,
       node.search_priority AS search_priority, score
ORDER BY score DESC, chunk_id ASC
""".strip()

    _SPARSE_QUERY = """
CALL db.index.fulltext.queryNodes('chunk_fulltext', $lucene_query, {limit: $top_k})
YIELD node, score
WHERE node:Chunk
  AND coalesce(node.searchable, false) = true
  AND coalesce(node.inactive, false) = false
  AND ($standard_id IS NULL OR node.standard_id = $standard_id)
  AND ($zone IS NULL OR node.zone = $zone)
RETURN node.chunk_id AS chunk_id, node.standard_id AS standard_id,
       node.zone AS zone, node.chunk_type AS chunk_type, node.text AS text,
       node.contextualized_text AS contextualized_text,
       node.citation_label AS citation_label,
       node.pdf_page_start AS pdf_page_start, node.pdf_page_end AS pdf_page_end,
       node.search_priority AS search_priority, score
ORDER BY score DESC, chunk_id ASC
""".strip()

    def __init__(self, driver: Any, openai_client: Any, *, database: str,
                 embedding_model: str, config: HybridConfig | None = None) -> None:
        self.driver = driver
        self.openai_client = openai_client
        self.database = database
        self.embedding_model = embedding_model
        self.config = config or HybridConfig()

    def _embed(self, question: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[question],
            dimensions=self.config.embedding_dimensions,
            encoding_format="float",
        )
        return list(response.data[0].embedding)

    @staticmethod
    def _rows(result: Any) -> list[dict[str, Any]]:
        return [record.data() if hasattr(record, "data") else dict(record) for record in result]

    def search(self, question: str, *, standard_id: str | None = None,
               zone: str | None = None) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        standard_id = normalize_standard_id(standard_id)
        embedding = self._embed(question)
        common = {"standard_id": standard_id, "zone": zone}
        with self.driver.session(database=self.database) as session:
            dense = self._rows(session.run(
                self._DENSE_QUERY, top_k=self.config.dense_top_k,
                embedding=embedding, **common,
            ))
            sparse = self._rows(session.run(
                self._SPARSE_QUERY, top_k=self.config.sparse_top_k,
                lucene_query=escape_lucene_query(question), **common,
            ))
        return reciprocal_rank_fusion(
            dense, sparse, rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            top_k=self.config.seed_top_k,
        )
