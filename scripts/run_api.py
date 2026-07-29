from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_ENV = (
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "OPENAI_API_KEY",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_CHAT_MODEL",
)


def missing_environment(environ: Mapping[str, str]) -> list[str]:
    return [name for name in REQUIRED_ENV if not environ.get(name, "").strip()]


def verify_neo4j() -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            session.run("RETURN 1 AS ready").single(strict=True)
    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Accounting RAG 로컬 웹 서버 실행")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    load_dotenv(PROJECT_ROOT / ".env")
    missing = missing_environment(os.environ)
    if missing:
        parser.error("missing environment variables: " + ", ".join(missing))

    try:
        verify_neo4j()
    except Exception as exc:
        parser.error(f"Neo4j connectivity check failed: {type(exc).__name__}")

    import uvicorn

    print(f"Accounting RAG: http://{args.host}:{args.port}")
    uvicorn.run(
        "accounting_rag.api.app:app",
        host=args.host,
        port=args.port,
        app_dir=str(SRC),
    )


if __name__ == "__main__":
    main()
