from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.graph.semantic_builder import build_semantic_kg, write_semantic_kg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an approved semantic KG from official definitions.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "semantic")
    args = parser.parse_args()
    report = write_semantic_kg(args.output_dir.resolve(), build_semantic_kg(args.processed_dir.resolve()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
