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

from accounting_rag.graph.loader import load_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Load structured K-IFRS data into Neo4j.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl")
    parser.add_argument("--schema", type=Path, default=PROJECT_ROOT / "db" / "schema.cypher")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install the project dependencies first.")

    load_dotenv(PROJECT_ROOT / ".env")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        report = load_graph(
            args.processed_dir.resolve(),
            args.chunks.resolve(),
            driver,
            os.environ["NEO4J_DATABASE"],
            args.schema.resolve(),
            args.batch_size,
        )
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
