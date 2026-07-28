from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.embeddings import (  # noqa: E402
    EmbeddingConfig,
    batches,
    read_jsonl,
    validate_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate embedding cache and Neo4j vector state.")
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "embedding.json")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [name for name in required if not os.getenv(name)]
    if not model:
        missing.append("OPENAI_EMBEDDING_MODEL")
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = EmbeddingConfig(model=model, dimensions=settings["dimensions"], batch_size=settings.get("batch_size", 64))
    cache_rows = read_jsonl(args.cache)
    cache_report = validate_cache(read_jsonl(args.chunks), cache_rows, config)

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            target = session.run(
                "MATCH (c:Chunk {searchable: true}) RETURN count(c) AS target"
            ).single()["target"]
            matched = valid = 0
            for group in batches(cache_rows, 500):
                state = session.run(
                    "UNWIND $rows AS row "
                    "MATCH (c:Chunk {chunk_id: row.chunk_id, searchable: true}) "
                    "RETURN count(c) AS matched, "
                    "count(CASE WHEN c.embedding IS NOT NULL "
                    "AND size(c.embedding) = $dimensions "
                    "AND c.embedding_model = $model "
                    "AND c.embedding_dimensions = $dimensions "
                    "AND c.embedding_text_sha256 = row.text_sha256 THEN 1 END) AS valid",
                    rows=[
                        {"chunk_id": row["chunk_id"], "text_sha256": row["text_sha256"]}
                        for row in group
                    ],
                    model=config.model,
                    dimensions=config.dimensions,
                ).single()
                matched += state["matched"]
                valid += state["valid"]
            index = session.run(
                "SHOW INDEXES YIELD name, state, type, options "
                "WHERE name = 'chunk_embedding_vector' "
                "RETURN name, state, type, options"
            ).single()
    finally:
        driver.close()
    index_dimensions = None
    if index:
        index_dimensions = (
            index["options"].get("indexConfig", {}).get("vector.dimensions")
        )
    neo4j_report = {
        "target": target,
        "matched_cache_records": matched,
        "valid": valid,
        "index_state": index["state"] if index else None,
        "index_type": index["type"] if index else None,
        "index_dimensions": index_dimensions,
    }
    report = {
        "valid": cache_report["valid"]
        and target == cache_report["expected"]
        and matched == cache_report["expected"]
        and valid == cache_report["expected"]
        and neo4j_report["index_state"] == "ONLINE"
        and index_dimensions == config.dimensions,
        "cache": cache_report,
        "neo4j": neo4j_report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
