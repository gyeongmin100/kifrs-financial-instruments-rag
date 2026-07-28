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

from accounting_rag.retrieval.graph_expansion import (  # noqa: E402
    GraphExpander,
    GraphExpansionConfig,
)
from accounting_rag.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from accounting_rag.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from accounting_rag.retrieval.reranker import OpenAIReranker, RerankConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid seed search + graph expansion + OpenAI reranking."
    )
    parser.add_argument("question")
    parser.add_argument("--standard-id")
    parser.add_argument("--zone")
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "retrieval.yaml"
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        from openai import OpenAI
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install project dependencies first.")

    load_dotenv(PROJECT_ROOT / ".env")
    required = (
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "OPENAI_API_KEY",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_RERANK_MODEL",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    config_path = args.config.resolve()
    try:
        pipeline = RetrievalPipeline(
            HybridRetriever(
                driver,
                client,
                database=os.environ["NEO4J_DATABASE"],
                embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
                config=HybridConfig.from_yaml(config_path),
            ),
            GraphExpander(
                driver,
                database=os.environ["NEO4J_DATABASE"],
                config=GraphExpansionConfig.from_yaml(config_path),
            ),
            OpenAIReranker(
                client,
                model=os.environ["OPENAI_RERANK_MODEL"],
                config=RerankConfig.from_yaml(config_path),
            ),
        )
        output = pipeline.retrieve(
            args.question,
            standard_id=args.standard_id,
            zone=args.zone,
            top_k=args.top_k,
        )
    finally:
        driver.close()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
