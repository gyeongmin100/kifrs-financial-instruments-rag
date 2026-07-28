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
from accounting_rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Hybrid seed search plus sibling completion, without answering."
    )
    parser.add_argument("question")
    parser.add_argument("--standard-id")
    parser.add_argument("--zone")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--full-text", action="store_true", help="본문을 자르지 않고 출력한다")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "retrieval.yaml"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
                "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        result = RetrievalPipeline(
            HybridRetriever(
                driver, client,
                database=os.environ["NEO4J_DATABASE"],
                embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
                config=HybridConfig.from_yaml(args.config),
            ),
            driver,
            database=os.environ["NEO4J_DATABASE"],
            config=RetrievalConfig.from_yaml(args.config),
        ).retrieve(
            args.question, standard_id=args.standard_id,
            zone=args.zone, top_k=args.top_k,
        )
    finally:
        driver.close()

    if not args.full_text:
        for item in result["results"]:
            for key in ("text", "contextualized_text"):
                value = item.get(key)
                if isinstance(value, str) and len(value) > 200:
                    item[key] = value[:200] + "…"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
