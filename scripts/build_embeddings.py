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
    build_embedding_cache,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable OpenAI chunk embedding cache.")
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "chunk_embeddings.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "embedding.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "embedding_manifest.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "data" / "embeddings" / "EMBEDDING_QUALITY_REPORT.md")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install the project dependencies first.")

    load_dotenv(PROJECT_ROOT / ".env")
    model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if not os.getenv("OPENAI_API_KEY") or not model:
        parser.error("OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL are required")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = EmbeddingConfig(
        model=model,
        dimensions=settings["dimensions"],
        batch_size=settings.get("batch_size", 64),
    )
    report = build_embedding_cache(read_jsonl(args.chunks), args.cache, OpenAI(), config)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = (
        "# Embedding Quality Report\n\n"
        f"- valid: `{str(report['valid']).lower()}`\n"
        f"- model: `{report['model']}`\n"
        f"- dimensions: `{report['dimensions']}`\n"
        f"- searchable chunks: `{report['searchable_chunks']}`\n"
        f"- reused from cache: `{report['cached_before']}`\n"
        f"- newly embedded: `{report['created']}`\n"
        f"- final cache records: `{report['cache_records']}`\n"
    )
    args.report.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
