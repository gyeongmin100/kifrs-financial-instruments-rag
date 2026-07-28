from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.pdf_page_mapper import (  # noqa: E402
    PageMappingConfig,
    read_jsonl,
    validate_page_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate K-IFRS PDF page mappings.")
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "page_mapping.json"
    )
    args = parser.parse_args()
    source = args.processed_dir.resolve()
    report = validate_page_mapping(
        read_jsonl(source / "paragraphs.jsonl"),
        read_jsonl(source / "blocks.jsonl"),
        read_jsonl(source / "tables.jsonl"),
        read_jsonl(source / "footnotes.jsonl"),
        read_jsonl(source / "pdf_pages.jsonl"),
        PageMappingConfig.from_json(args.config.resolve()),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
