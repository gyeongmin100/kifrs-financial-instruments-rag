from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .hwpx_parser import HwpxParser, parse_references


STANDARD_IDS = ("1032", "1039", "1107", "1109")
ZONE_RANK = {
    "standard_body": 0,
    "appendix_definitions": 1,
    "application_guidance": 2,
    "application_examples": 3,
    "implementation_guidance": 4,
    "basis_for_conclusions": 5,
    "dissenting_opinion": 6,
    "committee_resolution": 7,
    "ifrs_comparison": 8,
    "amendment_history": 9,
    "front_matter": 10,
}
NUMERIC_REFERENCE_CANDIDATE_RE = re.compile(r"문단\s*(?=(?:한|[A-Z]|\d))")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prefix(number: str) -> str:
    match = re.match(r"^(한|[A-Z]+\.?)", number)
    return match.group(0) if match else ""


def _candidate_numbers(number: str | None, start_number: str | None = None) -> list[str]:
    if not number:
        return []
    candidates = [number]
    inherited_prefix = _prefix(start_number or "")
    if inherited_prefix and not _prefix(number):
        candidates.append(inherited_prefix + number)
    return candidates


def _choose_single(candidates: list[dict], source: dict) -> tuple[str, list[str]]:
    if not candidates:
        return "unresolved_paragraph", []
    if len(candidates) == 1:
        return "resolved", [candidates[0]["paragraph_id"]]
    same_zone = [item for item in candidates if item["zone"] == source.get("source_zone")]
    if len(same_zone) == 1:
        return "resolved", [same_zone[0]["paragraph_id"]]
    return "ambiguous", [item["paragraph_id"] for item in candidates]


def _resolve_reference(
    reference: dict,
    source: dict,
    inventories: dict[str, dict],
) -> dict:
    target_standard = reference.get("target_standard")
    start_number = reference.get("target_paragraph_start")
    end_number = reference.get("target_paragraph_end")
    result = {
        "resolution_status": None,
        "resolved_target_ids": [],
        "resolved_start_number": None,
        "resolved_end_number": None,
        "range_expanded_count": 0,
    }

    if not start_number:
        result["resolution_status"] = (
            "resolved_standard" if target_standard in inventories else "external_standard"
        )
        return result
    if target_standard not in inventories:
        result["resolution_status"] = "external_standard"
        return result

    inventory = inventories[target_standard]
    by_number = inventory["by_number"]
    start_candidates = by_number.get(start_number, [])
    if not end_number:
        status, target_ids = _choose_single(start_candidates, source)
        result.update(
            resolution_status=status,
            resolved_target_ids=target_ids,
            resolved_start_number=start_number if target_ids else None,
            range_expanded_count=len(target_ids),
        )
        return result

    end_variants = _candidate_numbers(end_number, start_number)
    end_candidates = [
        paragraph
        for candidate_number in end_variants
        for paragraph in by_number.get(candidate_number, [])
    ]
    possible_pairs: list[tuple[dict, dict]] = []
    for start in start_candidates:
        for end in end_candidates:
            if start["zone"] == end["zone"] and start["document_order"] <= end["document_order"]:
                possible_pairs.append((start, end))
    if not possible_pairs:
        result["resolution_status"] = "unresolved_range"
        return result

    same_source_zone = [
        pair for pair in possible_pairs if pair[0]["zone"] == source.get("source_zone")
    ]
    if same_source_zone:
        possible_pairs = same_source_zone
    possible_pairs.sort(
        key=lambda pair: (
            ZONE_RANK.get(pair[0]["zone"], 99),
            pair[1]["document_order"] - pair[0]["document_order"],
        )
    )
    best_score = (
        ZONE_RANK.get(possible_pairs[0][0]["zone"], 99),
        possible_pairs[0][1]["document_order"] - possible_pairs[0][0]["document_order"],
    )
    best_pairs = [
        pair
        for pair in possible_pairs
        if (
            ZONE_RANK.get(pair[0]["zone"], 99),
            pair[1]["document_order"] - pair[0]["document_order"],
        )
        == best_score
    ]
    if len(best_pairs) != 1:
        result["resolution_status"] = "ambiguous_range"
        result["resolved_target_ids"] = sorted(
            {item["paragraph_id"] for pair in best_pairs for item in pair}
        )
        result["range_expanded_count"] = len(result["resolved_target_ids"])
        return result

    start, end = best_pairs[0]
    expanded = [
        paragraph["paragraph_id"]
        for paragraph in inventory["ordered"]
        if paragraph["zone"] == start["zone"]
        and start["document_order"] <= paragraph["document_order"] <= end["document_order"]
    ]
    result.update(
        resolution_status="resolved_range",
        resolved_target_ids=expanded,
        resolved_start_number=start["number"],
        resolved_end_number=end["number"],
        range_expanded_count=len(expanded),
    )
    return result


def _flatten_tables(document: dict, standard_id: str) -> list[dict]:
    tables: list[dict] = []
    paragraph_lookup = {
        paragraph["paragraph_id"]: paragraph for paragraph in document["paragraphs"]
    }
    for paragraph in document["paragraphs"]:
        for table in paragraph["tables"]:
            tables.append(
                {
                    "standard_id": standard_id,
                    "zone": paragraph["zone"],
                    "section_path": paragraph["section_path"],
                    "source_section": paragraph["source_section"],
                    **table,
                }
            )
    for orphan in document["orphan_tables"]:
        tables.append(
            {
                "standard_id": standard_id,
                "zone": orphan["zone"],
                "section_path": orphan["section_path"],
                "source_section": orphan["source_section"],
                **orphan["table"],
            }
        )
    return tables


def _flatten_footnotes(document: dict, standard_id: str) -> list[dict]:
    footnotes: list[dict] = []
    for paragraph in document["paragraphs"]:
        for footnote in paragraph["footnotes"]:
            footnotes.append(
                {
                    "standard_id": standard_id,
                    "zone": paragraph["zone"],
                    "section_path": paragraph["section_path"],
                    "source_section": paragraph["source_section"],
                    **footnote,
                }
            )
    for orphan in document["orphan_footnotes"]:
        footnotes.append(
            {
                "standard_id": standard_id,
                "zone": orphan["zone"],
                "section_path": orphan["section_path"],
                "source_section": orphan["source_section"],
                **orphan["footnote"],
            }
        )
    return footnotes


def _reference_sources(
    documents: dict[str, dict], tables: list[dict], footnotes: list[dict]
) -> tuple[list[dict], list[dict]]:
    sources: list[dict] = []
    unparsed: list[dict] = []
    for standard_id, document in documents.items():
        for block in document["blocks"]:
            if block["zone"] == "front_matter":
                continue
            block_references = block.pop("references")
            sources.append(
                {
                    "source_type": "block",
                    "source_id": block["block_id"],
                    "source_standard": standard_id,
                    "source_zone": block["zone"],
                    "source_paragraph_id": block["parent_paragraph_id"],
                    "text": block["text"],
                    "references": block_references,
                }
            )
            numeric_markers = list(NUMERIC_REFERENCE_CANDIDATE_RE.finditer(block["text"]))
            paragraph_references = [
                item for item in block_references if item["target_paragraph_start"]
            ]
            for marker in numeric_markers:
                if not any(
                    0 <= (item["char_start"] or 0) - marker.end() <= 3
                    for item in paragraph_references
                ):
                    unparsed.append(
                        {
                            "standard_id": standard_id,
                            "source_id": block["block_id"],
                            "zone": block["zone"],
                            "candidate": block["text"][marker.start() : marker.start() + 80],
                        }
                    )

    for table in tables:
        text = " ".join(cell["text"] for cell in table["cells"] if cell["text"])
        sources.append(
            {
                "source_type": "table",
                "source_id": table["table_id"],
                "source_standard": table["standard_id"],
                "source_zone": table["zone"],
                "source_paragraph_id": table["parent_paragraph_id"],
                "text": text,
                "references": [
                    asdict(reference)
                    for reference in parse_references(text, table["standard_id"])
                ],
            }
        )
    for footnote in footnotes:
        sources.append(
            {
                "source_type": "footnote",
                "source_id": footnote["footnote_id"],
                "source_standard": footnote["standard_id"],
                "source_zone": footnote["zone"],
                "source_paragraph_id": footnote["parent_paragraph_id"],
                "text": footnote["text"],
                "references": [
                    asdict(reference)
                    for reference in parse_references(
                        footnote["text"], footnote["standard_id"]
                    )
                ],
            }
        )
    return sources, unparsed


def build_dataset(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents: dict[str, dict] = {}
    source_paths: dict[str, Path] = {}
    for standard_id in STANDARD_IDS:
        path = source_dir / f"K-IFRS_{standard_id}.hwpx"
        if not path.exists():
            raise FileNotFoundError(path)
        documents[standard_id] = HwpxParser(standard_id).parse(path)
        source_paths[standard_id] = path

    paragraphs: list[dict] = []
    blocks: list[dict] = []
    tables: list[dict] = []
    footnotes: list[dict] = []
    inventories: dict[str, dict] = {}
    for standard_id, document in documents.items():
        normalized_paragraphs = []
        for paragraph in document["paragraphs"]:
            normalized = {
                key: value
                for key, value in paragraph.items()
                if key not in {"tables", "footnotes", "references"}
            }
            normalized["standard_id"] = standard_id
            normalized_paragraphs.append(normalized)
        paragraphs.extend(normalized_paragraphs)
        blocks.extend(
            {"standard_id": standard_id, **block} for block in document["blocks"]
        )
        tables.extend(_flatten_tables(document, standard_id))
        footnotes.extend(_flatten_footnotes(document, standard_id))
        by_number: dict[str, list[dict]] = defaultdict(list)
        for paragraph in normalized_paragraphs:
            by_number[paragraph["number"]].append(paragraph)
        inventories[standard_id] = {
            "ordered": normalized_paragraphs,
            "by_number": dict(by_number),
        }

    sources, unparsed = _reference_sources(documents, tables, footnotes)
    references: list[dict] = []
    reference_counter = 0
    for source in sources:
        for reference in source.pop("references"):
            reference_counter += 1
            row = {
                "reference_id": f"REF-{reference_counter:07d}",
                "source_type": source["source_type"],
                "source_id": source["source_id"],
                "source_standard": source["source_standard"],
                "source_zone": source["source_zone"],
                "source_paragraph_id": source["source_paragraph_id"],
                "reference_group_id": (
                    f"{source['source_id']}-G{reference['group_index']:03d}"
                    if reference["group_index"] is not None
                    else None
                ),
                **reference,
            }
            row.update(_resolve_reference(row, row, inventories))
            references.append(row)

    _write_jsonl(output_dir / "paragraphs.jsonl", paragraphs)
    _write_jsonl(output_dir / "blocks.jsonl", blocks)
    _write_jsonl(output_dir / "tables.jsonl", tables)
    _write_jsonl(output_dir / "footnotes.jsonl", footnotes)
    _write_jsonl(output_dir / "references.jsonl", references)
    _write_jsonl(output_dir / "unparsed_reference_candidates.jsonl", unparsed)
    unresolved_references = [
        row
        for row in references
        if row["resolution_status"]
        in {
            "unresolved_paragraph",
            "unresolved_range",
            "ambiguous",
            "ambiguous_range",
        }
    ]
    _write_jsonl(output_dir / "unresolved_references.jsonl", unresolved_references)

    manifest = {
        "standards": [],
        "totals": {
            "paragraphs": len(paragraphs),
            "blocks": len(blocks),
            "tables": len(tables),
            "footnotes": len(footnotes),
            "references": len(references),
            "unparsed_reference_candidates": len(unparsed),
        },
    }
    for standard_id in STANDARD_IDS:
        standard_paragraphs = [
            row for row in paragraphs if row["standard_id"] == standard_id
        ]
        manifest["standards"].append(
            {
                "standard_id": standard_id,
                "source_file": source_paths[standard_id].name,
                "source_sha256": _sha256(source_paths[standard_id]),
                "paragraphs": len(standard_paragraphs),
                "blocks": sum(1 for row in blocks if row["standard_id"] == standard_id),
                "tables": sum(1 for row in tables if row["standard_id"] == standard_id),
                "footnotes": sum(
                    1 for row in footnotes if row["standard_id"] == standard_id
                ),
                "references": sum(
                    1 for row in references if row["source_standard"] == standard_id
                ),
                "zones": dict(Counter(row["zone"] for row in standard_paragraphs)),
            }
        )
    with (output_dir / "document_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")

    status_counts = Counter(row["resolution_status"] for row in references)
    range_counts = Counter(
        row["range_delimiter"]
        for row in references
        if row["range_delimiter"] is not None
    )
    duplicate_counts = {
        standard_id: sum(
            1 for values in inventory["by_number"].values() if len(values) > 1
        )
        for standard_id, inventory in inventories.items()
    }
    lines = [
        "# 전체 기준서 파싱 품질보고서",
        "",
        "## 생성 결과",
        "",
        f"- 문단: {len(paragraphs):,}개",
        f"- 원문 블록: {len(blocks):,}개",
        f"- 표: {len(tables):,}개",
        f"- 각주: {len(footnotes):,}개",
        f"- 명시적 참조: {len(references):,}개",
        f"- 파싱되지 않은 숫자형 `문단` 후보: {len(unparsed):,}개",
        f"- 표현은 추출했지만 현재 판본에서 대상을 확정하지 못한 참조: {len(unresolved_references):,}개",
        "",
        "## 참조 해석 상태",
        "",
    ]
    lines.extend(f"- {status}: {count:,}개" for status, count in status_counts.most_common())
    lines.extend(
        [
            "",
            "## 범위 기호",
            "",
        ]
    )
    lines.extend(f"- `{delimiter}`: {count:,}개" for delimiter, count in range_counts.items())
    lines.extend(
        [
            "",
            "## 중복 문단번호",
            "",
        ]
    )
    lines.extend(
        f"- 제{standard_id}호: {count:,}개 번호가 둘 이상 존재"
        for standard_id, count in duplicate_counts.items()
    )
    lines.extend(
        [
            "",
            "## 판정 원칙",
            "",
            "- 일반 숫자 범위는 무시하고 `문단` 뒤의 유효한 번호 형식만 참조 후보로 인식합니다.",
            "- 범위는 대상 기준서의 실제 문단 순서와 같은 영역을 기준으로 펼칩니다.",
            "- 외부 기준서, 삭제 문단, 중복 번호 등 자동 확정할 수 없는 참조는 상태값으로 남깁니다.",
            "- 대상 미확정 참조도 원문과 시작·끝 번호를 `unresolved_references.jsonl`에 보존합니다.",
            "- 이 단계는 구조와 참조 검증이며 PDF 전체 페이지 매핑은 다음 단계입니다.",
            "",
        ]
    )
    (output_dir / "QUALITY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return manifest
