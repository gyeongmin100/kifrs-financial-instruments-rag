from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dimensions: int = 3072
    batch_size: int = 64

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Embedding model must not be empty")
        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if not 1 <= self.batch_size <= 2048:
            raise ValueError("Batch size must be between 1 and 2048")


def batches(rows: Sequence[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def searchable_inputs(chunks: Iterable[dict]) -> list[dict]:
    result = []
    seen = set()
    for chunk in chunks:
        if chunk.get("searchable") is not True:
            continue
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("contextualized_text")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Every searchable chunk must have a chunk_id")
        if chunk_id in seen:
            raise ValueError(f"Duplicate searchable chunk_id: {chunk_id}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty contextualized_text: {chunk_id}")
        seen.add(chunk_id)
        result.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "text_sha256": text_sha256(text),
            }
        )
    return result


def _valid_cached_record(record: dict, config: EmbeddingConfig) -> bool:
    vector = record.get("embedding")
    return (
        isinstance(record.get("chunk_id"), str)
        and record.get("model") == config.model
        and record.get("dimensions") == config.dimensions
        and isinstance(record.get("text_sha256"), str)
        and isinstance(vector, list)
        and len(vector) == config.dimensions
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector)
    )


def reusable_cache(cache_rows: Iterable[dict], config: EmbeddingConfig) -> dict[str, dict]:
    """Keep the latest structurally valid record per chunk for this model/dimension."""
    result = {}
    for row in cache_rows:
        if _valid_cached_record(row, config):
            result[row["chunk_id"]] = row
    return result


def embed_texts(client, texts: Sequence[str], config: EmbeddingConfig) -> list[list[float]]:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding input must be a non-empty array of non-empty strings")
    if len(texts) > 2048:
        raise ValueError("OpenAI embedding requests support at most 2048 inputs")
    response = client.embeddings.create(
        input=list(texts),
        model=config.model,
        dimensions=config.dimensions,
        encoding_format="float",
    )
    data = list(response.data)
    if len(data) != len(texts):
        raise ValueError(f"Embedding response count mismatch: {len(data)} != {len(texts)}")
    by_index = {}
    for item in data:
        index = item.index
        vector = list(item.embedding)
        if index in by_index or not 0 <= index < len(texts):
            raise ValueError(f"Invalid embedding response index: {index}")
        if len(vector) != config.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch at index {index}: "
                f"{len(vector)} != {config.dimensions}"
            )
        by_index[index] = vector
    if set(by_index) != set(range(len(texts))):
        raise ValueError("Embedding response indexes are incomplete")
    return [by_index[index] for index in range(len(texts))]


def _append_records(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())


def _rewrite_cache(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def build_embedding_cache(
    chunks: Iterable[dict],
    cache_path: Path,
    client,
    config: EmbeddingConfig,
) -> dict:
    inputs = searchable_inputs(chunks)
    cached = reusable_cache(read_jsonl(cache_path), config)
    pending = [
        row for row in inputs
        if row["chunk_id"] not in cached
        or cached[row["chunk_id"]]["text_sha256"] != row["text_sha256"]
    ]
    created = 0
    for group in batches(pending, config.batch_size):
        vectors = embed_texts(client, [row["text"] for row in group], config)
        records = [
            {
                "chunk_id": row["chunk_id"],
                "model": config.model,
                "dimensions": config.dimensions,
                "text_sha256": row["text_sha256"],
                "embedding": vector,
            }
            for row, vector in zip(group, vectors)
        ]
        _append_records(cache_path, records)
        cached.update({row["chunk_id"]: row for row in records})
        created += len(records)

    ordered = [cached[row["chunk_id"]] for row in inputs]
    _rewrite_cache(cache_path, ordered)
    return {
        "valid": len(ordered) == len(inputs),
        "model": config.model,
        "dimensions": config.dimensions,
        "batch_size": config.batch_size,
        "searchable_chunks": len(inputs),
        "cached_before": len(inputs) - len(pending),
        "created": created,
        "cache_records": len(ordered),
    }


def validate_cache(chunks: Iterable[dict], cache_rows: Iterable[dict], config: EmbeddingConfig) -> dict:
    inputs = searchable_inputs(chunks)
    expected = {row["chunk_id"]: row["text_sha256"] for row in inputs}
    rows = list(cache_rows)
    ids = [row.get("chunk_id") for row in rows]
    duplicates = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
    actual = {row.get("chunk_id"): row for row in rows}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    invalid = sorted(
        chunk_id for chunk_id, row in actual.items()
        if chunk_id in expected
        and (not _valid_cached_record(row, config) or row.get("text_sha256") != expected[chunk_id])
    )
    return {
        "valid": not duplicates and not missing and not extra and not invalid,
        "expected": len(expected),
        "actual": len(rows),
        "duplicates": duplicates,
        "missing": missing,
        "extra": extra,
        "invalid": invalid,
        "model": config.model,
        "dimensions": config.dimensions,
    }


def load_embeddings_neo4j(cache_rows: Sequence[dict], driver, database: str, batch_size: int = 500) -> dict:
    updated = 0
    with driver.session(database=database) as session:
        for group in batches(cache_rows, batch_size):
            record = session.run(
                "UNWIND $rows AS row "
                "MATCH (c:Chunk {chunk_id: row.chunk_id}) "
                "SET c.embedding = row.embedding, "
                "c.embedding_model = row.model, "
                "c.embedding_dimensions = row.dimensions, "
                "c.embedding_text_sha256 = row.text_sha256 "
                "RETURN count(c) AS updated",
                rows=list(group),
            ).single()
            updated += record["updated"]
    return {"cache_records": len(cache_rows), "updated_chunks": updated, "valid": updated == len(cache_rows)}
