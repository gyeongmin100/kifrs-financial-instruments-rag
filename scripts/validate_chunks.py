from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.chunk_builder import (
    ChunkingConfig,
    read_jsonl,
    validate_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate traceability and size limits of generated chunks."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "chunks",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "chunking.json",
    )
    args = parser.parse_args()
    processed_dir = args.processed_dir.resolve()
    chunks_dir = args.chunks_dir.resolve()
    config = ChunkingConfig.from_json(args.config.resolve())
    report = validate_chunks(
        read_jsonl(chunks_dir / "chunks.jsonl"),
        read_jsonl(processed_dir / "paragraphs.jsonl"),
        read_jsonl(processed_dir / "blocks.jsonl"),
        read_jsonl(processed_dir / "tables.jsonl"),
        read_jsonl(processed_dir / "footnotes.jsonl"),
        config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
