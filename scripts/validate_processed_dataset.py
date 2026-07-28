from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def require_unique(rows: list[dict], key: str, label: str) -> set[str]:
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} IDs")
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed K-IFRS JSONL files.")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()

    paragraphs = read_jsonl(data_dir / "paragraphs.jsonl")
    blocks = read_jsonl(data_dir / "blocks.jsonl")
    tables = read_jsonl(data_dir / "tables.jsonl")
    footnotes = read_jsonl(data_dir / "footnotes.jsonl")
    references = read_jsonl(data_dir / "references.jsonl")
    manifest = json.loads((data_dir / "document_manifest.json").read_text("utf-8"))

    paragraph_ids = require_unique(paragraphs, "paragraph_id", "paragraph")
    block_ids = require_unique(blocks, "block_id", "block")
    table_ids = require_unique(tables, "table_id", "table")
    footnote_ids = require_unique(footnotes, "footnote_id", "footnote")
    reference_ids = require_unique(references, "reference_id", "reference")

    for paragraph in paragraphs:
        missing = set(paragraph["block_ids"]) - block_ids
        if missing:
            raise ValueError(f"Missing blocks for {paragraph['paragraph_id']}: {missing}")
    for row in [*blocks, *tables, *footnotes]:
        parent = row.get("parent_paragraph_id")
        if parent and parent not in paragraph_ids:
            raise ValueError(f"Missing parent paragraph {parent}")
    source_ids = {"block": block_ids, "table": table_ids, "footnote": footnote_ids}
    for reference in references:
        if reference["source_id"] not in source_ids[reference["source_type"]]:
            raise ValueError(f"Missing source for {reference['reference_id']}")
        missing_targets = set(reference["resolved_target_ids"]) - paragraph_ids
        if missing_targets:
            raise ValueError(
                f"Missing resolved targets for {reference['reference_id']}: {missing_targets}"
            )
        if reference["range_expanded_count"] != len(reference["resolved_target_ids"]):
            raise ValueError(f"Range count mismatch for {reference['reference_id']}")

    actual_counts = {
        "paragraphs": len(paragraph_ids),
        "blocks": len(block_ids),
        "tables": len(table_ids),
        "footnotes": len(footnote_ids),
        "references": len(reference_ids),
        "unparsed_reference_candidates": len(
            read_jsonl(data_dir / "unparsed_reference_candidates.jsonl")
        ),
    }
    if actual_counts != manifest["totals"]:
        raise ValueError(
            f"Manifest totals differ: expected={manifest['totals']} actual={actual_counts}"
        )
    print(json.dumps({"valid": True, **actual_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
