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

from accounting_rag.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense + Sparse + weighted RRF search.")
    parser.add_argument("question")
    parser.add_argument("--standard-id")
    parser.add_argument("--zone")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config" / "retrieval.yaml")
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        from openai import OpenAI
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install project dependencies first.")
    load_dotenv(PROJECT_ROOT / ".env")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
                "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        retriever = HybridRetriever(
            driver, OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            database=os.environ["NEO4J_DATABASE"],
            embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
            config=HybridConfig.from_yaml(args.config.resolve()),
        )
        results = retriever.search(args.question, standard_id=args.standard_id, zone=args.zone)
    finally:
        driver.close()
    print(json.dumps({
        "question": args.question,
        "filters": {"standard_id": args.standard_id, "zone": args.zone},
        "result_count": len(results), "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
