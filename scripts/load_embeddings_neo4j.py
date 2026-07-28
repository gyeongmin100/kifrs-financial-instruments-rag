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

from accounting_rag.retrieval.embeddings import load_embeddings_neo4j, read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load cached embeddings onto Neo4j Chunk nodes.")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.jsonl")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(PROJECT_ROOT / ".env")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")
    rows = read_jsonl(args.cache)
    if not rows:
        parser.error(f"Embedding cache is empty or missing: {args.cache}")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        report = load_embeddings_neo4j(
            rows, driver, os.environ["NEO4J_DATABASE"], args.batch_size
        )
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
