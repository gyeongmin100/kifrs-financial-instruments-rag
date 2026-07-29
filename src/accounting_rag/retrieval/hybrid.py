from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
from hashlib import sha256
from typing import Any, Mapping, Sequence

import yaml


score_logger = logging.getLogger("accounting_rag.retrieval.scores")


@dataclass(frozen=True)
class HybridConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    seed_top_k: int = 6
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    embedding_dimensions: int = 3072
    dense_min_score: float | None = None
    sparse_min_score: float | None = None

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
        for name in ("dense_min_score", "sparse_min_score"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite or null")

    @classmethod
    def from_yaml(cls, path: Path) -> "HybridConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = raw.get("hybrid", raw)
        return cls(
            dense_top_k=int(values.get("dense_top_k", 50)),
            sparse_top_k=int(values.get("sparse_top_k", 50)),
            seed_top_k=int(values.get("seed_top_k", 6)),
            rrf_k=int(values.get("rrf_k", 60)),
            dense_weight=float(values.get("dense_weight", 1.0)),
            sparse_weight=float(values.get("sparse_weight", 1.0)),
            embedding_dimensions=int(values.get("embedding_dimensions", 3072)),
            dense_min_score=(
                None if values.get("dense_min_score") is None
                else float(values["dense_min_score"])
            ),
            sparse_min_score=(
                None if values.get("sparse_min_score") is None
                else float(values["sparse_min_score"])
            ),
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

    def search_with_scores(self, question: str, *, standard_id: str | None = None,
                           zone: str | None = None,
                           sparse_query: str | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        sparse_query = (sparse_query or question).strip()
        if not sparse_query:
            raise ValueError("sparse_query must not be empty")
        standard_id = normalize_standard_id(standard_id)
        embedding = self._embed(question)
        common = {"standard_id": standard_id, "zone": zone}
        with self.driver.session(database=self.database) as session:
            raw_dense = self._rows(session.run(
                self._DENSE_QUERY, top_k=self.config.dense_top_k,
                embedding=embedding, **common,
            ))
            raw_sparse = self._rows(session.run(
                self._SPARSE_QUERY, top_k=self.config.sparse_top_k,
                lucene_query=escape_lucene_query(sparse_query), **common,
            ))
        dense = [
            row for row in raw_dense
            if self.config.dense_min_score is None
            or float(row["score"]) >= self.config.dense_min_score
        ]
        sparse = [
            row for row in raw_sparse
            if self.config.sparse_min_score is None
            or float(row["score"]) >= self.config.sparse_min_score
        ]
        fused = reciprocal_rank_fusion(
            dense, sparse, rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            top_k=self.config.seed_top_k,
        )
        if os.getenv("RETRIEVAL_SCORE_LOG") == "1":
            event: dict[str, Any] = {
                "event": "retrieval_scores",
                "question_hash": sha256(question.encode("utf-8")).hexdigest(),
                "filters": {"standard_id": standard_id, "zone": zone},
                "thresholds": {
                    "dense": self.config.dense_min_score,
                    "sparse": self.config.sparse_min_score,
                },
                "dense": [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "rank": rank,
                        "score": float(row["score"]),
                        "passed": row in dense,
                    }
                    for rank, row in enumerate(raw_dense, start=1)
                ],
                "sparse": [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "rank": rank,
                        "score": float(row["score"]),
                        "passed": row in sparse,
                    }
                    for rank, row in enumerate(raw_sparse, start=1)
                ],
                "selected_chunk_ids": [row["chunk_id"] for row in fused],
            }
            if os.getenv("RETRIEVAL_SCORE_LOG_INCLUDE_QUESTION") == "1":
                event["question"] = question
            score_logger.info(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        return {"dense": raw_dense, "sparse": raw_sparse, "results": fused}

    def search(self, question: str, *, standard_id: str | None = None,
               zone: str | None = None,
               sparse_query: str | None = None) -> list[dict[str, Any]]:
        return self.search_with_scores(
            question, standard_id=standard_id, zone=zone,
            sparse_query=sparse_query,
        )["results"]
