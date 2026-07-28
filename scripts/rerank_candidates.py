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

from accounting_rag.retrieval.reranker import OpenAIReranker, RerankConfig  # noqa: E402


def _read_candidates(path: str) -> list[dict]:
    if path == "-":
        payload = json.load(sys.stdin)
    else:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("candidates"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("input must be a JSON array or an object with results/candidates")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank hybrid/graph evidence candidates with OpenAI Structured Output."
    )
    parser.add_argument("question")
    parser.add_argument("--input", default="-", help="Candidate JSON path; default: stdin")
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "retrieval.yaml"
    )
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install project dependencies first.")
    load_dotenv(PROJECT_ROOT / ".env")
    missing = [name for name in ("OPENAI_API_KEY", "OPENAI_RERANK_MODEL") if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")
    try:
        candidates = _read_candidates(args.input)
        config = RerankConfig.from_yaml(args.config.resolve())
        results = OpenAIReranker(
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            model=os.environ["OPENAI_RERANK_MODEL"],
            config=config,
        ).rerank(args.question, candidates, top_k=args.top_k)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "question": args.question,
        "input_count": len(candidates),
        "result_count": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
