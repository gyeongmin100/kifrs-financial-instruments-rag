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
    build_page_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map parsed K-IFRS records to physical PDF pages."
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--pdf-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "page_mapping.json"
    )
    args = parser.parse_args()
    report = build_page_mapping(
        args.processed_dir.resolve(),
        args.pdf_dir.resolve(),
        PageMappingConfig.from_json(args.config.resolve()),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
