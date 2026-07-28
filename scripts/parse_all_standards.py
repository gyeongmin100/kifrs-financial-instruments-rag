from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.dataset_builder import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse four K-IFRS financial-instrument standards into JSONL."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "standards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    args = parser.parse_args()
    manifest = build_dataset(args.source_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(manifest["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
