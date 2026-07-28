from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "1.0"
ZONE_SEARCH_PRIORITY = {
    "standard_body": 3,
    "appendix_definitions": 3,
    "application_guidance": 3,
    "application_examples": 2,
    "implementation_guidance": 2,
    "basis_for_conclusions": 1,
    "dissenting_opinion": 1,
}
INACTIVE_RE = re.compile(r"삭제함|아직 시행되지 않는 .*반영하지 않음")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。])\s+|\n+")


@dataclass(frozen=True)
class ChunkingConfig:
    paragraph_target_chars: int = 900
    paragraph_max_chars: int = 1200
    table_target_chars: int = 1200
    table_max_chars: int = 1800
    repeated_intro_chars: int = 240
    footnote_context_chars: int = 600

    @classmethod
    def from_json(cls, path: Path) -> "ChunkingConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_long_text(text: str, max_chars: int) -> list[str]:
    text = _normalize(text)
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            remainder = sentence
            while len(remainder) > max_chars:
                split_at = remainder.rfind(" ", 0, max_chars + 1)
                if split_at < max_chars // 2:
                    split_at = max_chars
                parts.append(remainder[:split_at].strip())
                remainder = remainder[split_at:].strip()
            current = remainder
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _pack_units(units: list[dict], target_chars: int, max_chars: int) -> list[dict]:
    expanded: list[dict] = []
    for unit in units:
        for part in _split_long_text(unit["text"], max_chars):
            expanded.append({**unit, "text": part})

    groups: list[dict] = []
    current: dict | None = None
    for unit in expanded:
        if current is None:
            current = {
                "texts": [unit["text"]],
                "block_ids": list(unit["block_ids"]),
                "subparagraph_ids": list(unit["subparagraph_ids"]),
            }
            continue
        candidate_length = len("\n".join([*current["texts"], unit["text"]]))
        if candidate_length > target_chars:
            groups.append(current)
            current = {
                "texts": [unit["text"]],
                "block_ids": list(unit["block_ids"]),
                "subparagraph_ids": list(unit["subparagraph_ids"]),
            }
        else:
            current["texts"].append(unit["text"])
            current["block_ids"].extend(unit["block_ids"])
            current["subparagraph_ids"].extend(unit["subparagraph_ids"])
    if current is not None:
        groups.append(current)

    for group in groups:
        group["text"] = "\n".join(group.pop("texts"))
        group["block_ids"] = list(dict.fromkeys(group["block_ids"]))
        group["subparagraph_ids"] = list(dict.fromkeys(group["subparagraph_ids"]))
        if len(group["text"]) > max_chars:
            raise ValueError("Packed chunk exceeds maximum size")
    return groups


def _paragraph_units(paragraph: dict, block_lookup: dict[str, dict]) -> list[dict]:
    units: list[dict] = []
    subparagraph_index = 0
    current_subparagraph_id: str | None = None
    for block_id in paragraph["block_ids"]:
        block = block_lookup[block_id]
        text = _normalize(block["text"])
        if block["block_type"] == "paragraph":
            number = paragraph["number"]
            if text == number:
                text = ""
            elif text.startswith(number + " "):
                text = text[len(number) + 1 :].strip()
            current_subparagraph_id = None
        elif block["block_type"] == "subitem":
            subparagraph_index += 1
            current_subparagraph_id = (
                f"{paragraph['paragraph_id']}-S{subparagraph_index:02d}"
            )
        if not text:
            continue
        source_subparagraphs = (
            [current_subparagraph_id] if current_subparagraph_id else []
        )
        units.append(
            {
                "text": text,
                "block_ids": [block_id],
                "subparagraph_ids": source_subparagraphs,
            }
        )
    return units


def _citation_label(paragraph: dict | None, standard_id: str) -> str:
    if paragraph:
        return f"K-IFRS 제{standard_id}호 문단 {paragraph['number']}"
    return f"K-IFRS 제{standard_id}호"


def _context_header(
    standard_id: str, zone: str, section_path: list[str], paragraph: dict | None
) -> str:
    parts = [f"K-IFRS 제{standard_id}호", zone]
    if section_path:
        parts.append(" > ".join(section_path))
    if paragraph:
        parts.append(f"문단 {paragraph['number']}")
    return "[" + " | ".join(parts) + "]"


def _footnote_context(footnotes: list[dict], max_chars: int) -> str:
    text = "\n".join(
        f"각주 {item.get('number') or ''}: {item['text']}".strip()
        for item in footnotes
        if item.get("text")
    )
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _paragraph_chunks(
    paragraph: dict,
    block_lookup: dict[str, dict],
    tables: list[dict],
    footnotes: list[dict],
    source_sha256: str | None,
    config: ChunkingConfig,
) -> list[dict]:
    units = _paragraph_units(paragraph, block_lookup)
    if not units:
        return []
    full_text = "\n".join(unit["text"] for unit in units)
    if len(full_text) <= config.paragraph_max_chars:
        groups = [
            {
                "text": full_text,
                "block_ids": list(
                    dict.fromkeys(
                        block_id for unit in units for block_id in unit["block_ids"]
                    )
                ),
                "subparagraph_ids": list(
                    dict.fromkeys(
                        sub_id
                        for unit in units
                        for sub_id in unit["subparagraph_ids"]
                    )
                ),
            }
        ]
    else:
        groups = _pack_units(
            units,
            config.paragraph_target_chars,
            config.paragraph_max_chars,
        )

    priority = ZONE_SEARCH_PRIORITY.get(paragraph["zone"], 0)
    inactive = bool(INACTIVE_RE.search(full_text))
    intro = units[0]["text"][: config.repeated_intro_chars]
    footnote_text = _footnote_context(footnotes, config.footnote_context_chars)
    chunks: list[dict] = []
    for part_index, group in enumerate(groups, start=1):
        source_blocks = [block_lookup[block_id] for block_id in group["block_ids"]]
        mapped_blocks = [
            block for block in source_blocks if block.get("pdf_page_start") is not None
        ]
        page_start = (
            min(block["pdf_page_start"] for block in mapped_blocks)
            if mapped_blocks
            else paragraph.get("pdf_page_start")
        )
        page_end = (
            max(block["pdf_page_end"] for block in mapped_blocks)
            if mapped_blocks
            else paragraph.get("pdf_page_end")
        )
        page_confidence = (
            min(block["page_match_confidence"] for block in mapped_blocks)
            if mapped_blocks
            else paragraph.get("page_match_confidence", 0.0)
        )
        if paragraph.get("page_match_method") == "inferred_next_paragraph":
            paragraph_start = paragraph.get("pdf_page_start")
            paragraph_end = paragraph.get("pdf_page_end")
            if paragraph_start is not None:
                page_start = min(page_start, paragraph_start) if page_start else paragraph_start
                page_end = max(page_end, paragraph_end) if page_end else paragraph_end
                page_confidence = min(
                    page_confidence, paragraph.get("page_match_confidence", 0.0)
                )
        context_parts = [
            _context_header(
                paragraph["standard_id"],
                paragraph["zone"],
                paragraph["section_path"],
                paragraph,
            )
        ]
        if part_index > 1 and intro and intro not in group["text"]:
            context_parts.append(f"문단 도입: {intro}")
        context_parts.append(group["text"])
        if footnote_text:
            context_parts.append(footnote_text)
        contextualized_text = "\n".join(context_parts)
        chunks.append(
            {
                "schema_version": SCHEMA_VERSION,
                "chunk_id": f"{paragraph['paragraph_id']}-C{part_index:02d}",
                "chunk_type": "paragraph" if len(groups) == 1 else "paragraph_part",
                "text": group["text"],
                "contextualized_text": contextualized_text,
                "standard_id": paragraph["standard_id"],
                "zone": paragraph["zone"],
                "section_path": paragraph["section_path"],
                "document_order": paragraph["document_order"],
                "source_paragraph_ids": [paragraph["paragraph_id"]],
                "source_subparagraph_ids": group["subparagraph_ids"],
                "block_ids": group["block_ids"],
                "table_ids": [table["table_id"] for table in tables],
                "footnote_ids": [item["footnote_id"] for item in footnotes],
                "citation_label": _citation_label(
                    paragraph, paragraph["standard_id"]
                ),
                "pdf_page_start": page_start,
                "pdf_page_end": page_end,
                "page_match_confidence": page_confidence,
                "source_sha256": source_sha256,
                "part_index": part_index,
                "part_count": len(groups),
                "char_count": len(group["text"]),
                "contextualized_char_count": len(contextualized_text),
                "word_count": len(group["text"].split()),
                "search_priority": priority,
                "searchable": priority > 0 and not inactive,
                "inactive": inactive,
                "inactive_reason": "deleted_or_not_effective" if inactive else None,
            }
        )
    return chunks


def _table_rows(table: dict) -> list[str]:
    rows: dict[int, list[dict]] = defaultdict(list)
    for cell in table["cells"]:
        if cell.get("text"):
            rows[cell["row"]].append(cell)
    result: list[str] = []
    for row_number in sorted(rows):
        cells = sorted(rows[row_number], key=lambda item: item["column"])
        values = []
        for cell in cells:
            span = ""
            if cell.get("row_span", 1) > 1 or cell.get("column_span", 1) > 1:
                span = (
                    f" (행병합 {cell.get('row_span', 1)}, "
                    f"열병합 {cell.get('column_span', 1)})"
                )
            values.append(
                f"열 {cell['column'] + 1}{span}: {_normalize(cell['text'])}"
            )
        result.append(f"행 {row_number + 1}: " + " | ".join(values))
    return result


def _table_chunks(
    table: dict,
    paragraph: dict | None,
    block_ids: list[str],
    source_sha256: str | None,
    config: ChunkingConfig,
) -> list[dict]:
    row_units = [
        {"text": row, "block_ids": block_ids, "subparagraph_ids": []}
        for row in _table_rows(table)
    ]
    if not row_units:
        return []
    groups = _pack_units(
        row_units,
        config.table_target_chars,
        config.table_max_chars,
    )
    priority = ZONE_SEARCH_PRIORITY.get(table["zone"], 0)
    parent_inactive = bool(
        paragraph and INACTIVE_RE.search(paragraph.get("text", ""))
    )
    header_row = row_units[0]["text"] if table.get("repeat_header") else ""
    chunks: list[dict] = []
    for part_index, group in enumerate(groups, start=1):
        text = group["text"]
        if (
            part_index > 1
            and header_row
            and header_row not in text
            and len(header_row) + 1 + len(text) <= config.table_max_chars
        ):
            text = f"{header_row}\n{text}"
        context_parts = [
            _context_header(
                table["standard_id"],
                table["zone"],
                table["section_path"],
                paragraph,
            ),
            f"표 {table['table_id']}",
        ]
        if paragraph and paragraph.get("text"):
            context_parts.append(
                f"관련 문단 도입: {paragraph['text'][:config.repeated_intro_chars]}"
            )
        context_parts.append(text)
        contextualized_text = "\n".join(context_parts)
        substantive = (
            table.get("rows", 0) > 1
            or table.get("columns", 0) > 1
            or len(text) >= 40
        )
        chunks.append(
            {
                "schema_version": SCHEMA_VERSION,
                "chunk_id": f"{table['table_id']}-C{part_index:02d}",
                "chunk_type": "table",
                "text": text,
                "contextualized_text": contextualized_text,
                "standard_id": table["standard_id"],
                "zone": table["zone"],
                "section_path": table["section_path"],
                "document_order": (
                    paragraph["document_order"] if paragraph else None
                ),
                "source_paragraph_ids": (
                    [paragraph["paragraph_id"]] if paragraph else []
                ),
                "source_subparagraph_ids": [],
                "block_ids": block_ids,
                "table_ids": [table["table_id"]],
                "footnote_ids": [],
                "citation_label": _citation_label(
                    paragraph, table["standard_id"]
                ),
                "pdf_page_start": table.get("pdf_page_start") or (
                    paragraph.get("pdf_page_start") if paragraph else None
                ),
                "pdf_page_end": table.get("pdf_page_end") or (
                    paragraph.get("pdf_page_end") if paragraph else None
                ),
                "page_match_confidence": (
                    table.get("page_match_confidence")
                    if table.get("pdf_page_start") is not None
                    else (
                        paragraph.get("page_match_confidence", 0.0)
                        if paragraph
                        else 0.0
                    )
                ),
                "source_sha256": source_sha256,
                "part_index": part_index,
                "part_count": len(groups),
                "char_count": len(text),
                "contextualized_char_count": len(contextualized_text),
                "word_count": len(text.split()),
                "search_priority": priority,
                "searchable": priority > 0 and substantive and not parent_inactive,
                "inactive": parent_inactive,
                "inactive_reason": (
                    "deleted_or_not_effective"
                    if parent_inactive
                    else (None if substantive else "non_substantive_table")
                ),
            }
        )
    return chunks


def _subparagraph_ids(paragraphs: list[dict]) -> set[str]:
    return {
        f"{paragraph['paragraph_id']}-S{index:02d}"
        for paragraph in paragraphs
        for index in range(1, len(paragraph["subitems"]) + 1)
    }


def validate_chunks(
    chunks: list[dict],
    paragraphs: list[dict],
    blocks: list[dict],
    tables: list[dict],
    footnotes: list[dict],
    config: ChunkingConfig,
) -> dict:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs")

    paragraph_ids = {row["paragraph_id"] for row in paragraphs}
    block_ids = {row["block_id"] for row in blocks}
    table_ids = {row["table_id"] for row in tables}
    footnote_ids = {row["footnote_id"] for row in footnotes}
    subparagraph_ids = _subparagraph_ids(paragraphs)
    for chunk in chunks:
        if not chunk["text"].strip():
            raise ValueError(f"Empty chunk: {chunk['chunk_id']}")
        if chunk["char_count"] != len(chunk["text"]):
            raise ValueError(f"Character count mismatch: {chunk['chunk_id']}")
        missing = set(chunk["source_paragraph_ids"]) - paragraph_ids
        if missing:
            raise ValueError(f"Missing paragraph source: {chunk['chunk_id']} {missing}")
        missing = set(chunk["source_subparagraph_ids"]) - subparagraph_ids
        if missing:
            raise ValueError(
                f"Missing subparagraph source: {chunk['chunk_id']} {missing}"
            )
        if set(chunk["block_ids"]) - block_ids:
            raise ValueError(f"Missing block source: {chunk['chunk_id']}")
        if set(chunk["table_ids"]) - table_ids:
            raise ValueError(f"Missing table source: {chunk['chunk_id']}")
        if set(chunk["footnote_ids"]) - footnote_ids:
            raise ValueError(f"Missing footnote source: {chunk['chunk_id']}")
        maximum = (
            config.table_max_chars
            if chunk["chunk_type"] == "table"
            else config.paragraph_max_chars
        )
        if chunk["char_count"] > maximum:
            raise ValueError(f"Oversized chunk: {chunk['chunk_id']}")

    paragraph_chunk_sources = {
        source_id
        for chunk in chunks
        if chunk["chunk_type"] != "table"
        for source_id in chunk["source_paragraph_ids"]
    }
    expected_paragraph_sources = {
        paragraph["paragraph_id"]
        for paragraph in paragraphs
        if any(
            block["text"].strip()
            for block in blocks
            if block.get("parent_paragraph_id") == paragraph["paragraph_id"]
        )
    }
    missing_paragraphs = expected_paragraph_sources - paragraph_chunk_sources
    if missing_paragraphs:
        raise ValueError(f"Paragraphs without chunks: {sorted(missing_paragraphs)[:10]}")

    block_lookup = {row["block_id"]: row for row in blocks}
    expected_content_block_ids = {
        block_id
        for paragraph in paragraphs
        for unit in _paragraph_units(paragraph, block_lookup)
        for block_id in unit["block_ids"]
    }
    chunk_content_block_ids = {
        block_id
        for chunk in chunks
        if chunk["chunk_type"] != "table"
        for block_id in chunk["block_ids"]
    }
    missing_blocks = expected_content_block_ids - chunk_content_block_ids
    if missing_blocks:
        raise ValueError(f"Content blocks without chunks: {sorted(missing_blocks)[:10]}")

    chunk_subparagraph_ids = {
        source_id
        for chunk in chunks
        if chunk["chunk_type"] != "table"
        for source_id in chunk["source_subparagraph_ids"]
    }
    missing_subparagraphs = subparagraph_ids - chunk_subparagraph_ids
    if missing_subparagraphs:
        raise ValueError(
            f"Subparagraphs without chunks: {sorted(missing_subparagraphs)[:10]}"
        )

    chunk_footnote_ids = {
        footnote_id
        for chunk in chunks
        if chunk["chunk_type"] != "table"
        for footnote_id in chunk["footnote_ids"]
    }
    expected_footnote_ids = {
        footnote["footnote_id"]
        for footnote in footnotes
        if footnote.get("parent_paragraph_id") in paragraph_chunk_sources
    }
    missing_footnotes = expected_footnote_ids - chunk_footnote_ids
    if missing_footnotes:
        raise ValueError(f"Footnotes without chunks: {sorted(missing_footnotes)[:10]}")

    table_chunk_sources = {
        table_id
        for chunk in chunks
        if chunk["chunk_type"] == "table"
        for table_id in chunk["table_ids"]
    }
    expected_table_sources = {
        table["table_id"]
        for table in tables
        if any(cell.get("text", "").strip() for cell in table["cells"])
    }
    missing_tables = expected_table_sources - table_chunk_sources
    if missing_tables:
        raise ValueError(f"Tables without chunks: {sorted(missing_tables)[:10]}")

    return {
        "valid": True,
        "chunks": len(chunks),
        "chunk_types": dict(Counter(row["chunk_type"] for row in chunks)),
        "zones": dict(Counter(row["zone"] for row in chunks)),
        "searchable": sum(row["searchable"] for row in chunks),
        "inactive": sum(row["inactive"] for row in chunks),
        "source_paragraphs": len(paragraph_chunk_sources),
        "source_subparagraphs": len(chunk_subparagraph_ids),
        "source_content_blocks": len(chunk_content_block_ids),
        "source_tables": len(table_chunk_sources),
        "source_footnotes": len(chunk_footnote_ids),
        "missing_pdf_page": sum(
            row["pdf_page_start"] is None for row in chunks if row["searchable"]
        ),
        "max_char_count": max(row["char_count"] for row in chunks),
        "max_contextualized_char_count": max(
            row["contextualized_char_count"] for row in chunks
        ),
    }


def _quality_report(report: dict, config: ChunkingConfig) -> str:
    lines = [
        "# 검색용 Chunk 품질보고서",
        "",
        "## 설정",
        "",
        f"- 문단 목표/최대 길이: {config.paragraph_target_chars:,}/{config.paragraph_max_chars:,}자",
        f"- 표 목표/최대 길이: {config.table_target_chars:,}/{config.table_max_chars:,}자",
        "- 짧은 문단은 자동 병합하지 않음",
        "- 긴 문단은 원문 Block·Subparagraph 경계에서 분할",
        "- 표는 행 경계에서 분할하고 가능한 경우 첫 행을 반복",
        "",
        "## 생성 결과",
        "",
        f"- 전체 Chunk: {report['chunks']:,}개",
        f"- 검색 사용 Chunk: {report['searchable']:,}개",
        f"- 삭제·미시행 문단 Chunk: {report['inactive']:,}개",
        f"- 연결된 원문 문단: {report['source_paragraphs']:,}개",
        f"- 연결된 소분류: {report['source_subparagraphs']:,}개",
        f"- 연결된 내용 Block: {report['source_content_blocks']:,}개",
        f"- 연결된 표: {report['source_tables']:,}개",
        f"- 연결된 각주: {report['source_footnotes']:,}개",
        f"- PDF 페이지 미매핑 검색 Chunk: {report['missing_pdf_page']:,}개",
        f"- 최대 원문 길이: {report['max_char_count']:,}자",
        f"- 최대 문맥 포함 길이: {report['max_contextualized_char_count']:,}자",
        "",
        "## 유형별",
        "",
    ]
    lines.extend(
        f"- {chunk_type}: {count:,}개"
        for chunk_type, count in sorted(report["chunk_types"].items())
    )
    lines.extend(["", "## 영역별", ""])
    lines.extend(
        f"- {zone}: {count:,}개"
        for zone, count in sorted(report["zones"].items())
    )
    lines.extend(
        [
            "",
            "## 검증 결과",
            "",
            "- Chunk ID 중복 없음",
            "- 빈 Chunk 없음",
            "- 내용이 있는 모든 Paragraph/Subparagraph/continuation Block/Table/소속 Footnote 역추적 성공",
            "- 문단·표 최대 길이 제한 위반 없음",
            "- 내용이 있는 모든 문단과 표에 하나 이상의 Chunk 존재",
            "",
            "## 다음 확인사항",
            "",
            "- 현재 전체 데이터의 PDF 페이지가 아직 매핑되지 않은 경우 page 필드는 null로 유지합니다.",
            "- Neo4j 적재 전 Chunk 표본을 원문과 대조하고 PDF 페이지 매핑을 별도 완료합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def build_chunks(
    processed_dir: Path,
    output_dir: Path,
    config: ChunkingConfig | None = None,
) -> dict:
    config = config or ChunkingConfig()
    paragraphs = read_jsonl(processed_dir / "paragraphs.jsonl")
    blocks = read_jsonl(processed_dir / "blocks.jsonl")
    tables = read_jsonl(processed_dir / "tables.jsonl")
    footnotes = read_jsonl(processed_dir / "footnotes.jsonl")
    manifest = json.loads(
        (processed_dir / "document_manifest.json").read_text(encoding="utf-8")
    )

    paragraph_lookup = {row["paragraph_id"]: row for row in paragraphs}
    block_lookup = {row["block_id"]: row for row in blocks}
    tables_by_paragraph: dict[str, list[dict]] = defaultdict(list)
    footnotes_by_paragraph: dict[str, list[dict]] = defaultdict(list)
    table_block_ids: dict[str, list[str]] = defaultdict(list)
    for table in tables:
        if table.get("parent_paragraph_id"):
            tables_by_paragraph[table["parent_paragraph_id"]].append(table)
    for footnote in footnotes:
        if footnote.get("parent_paragraph_id"):
            footnotes_by_paragraph[footnote["parent_paragraph_id"]].append(footnote)
    for block in blocks:
        for table_id in block["table_ids"]:
            table_block_ids[table_id].append(block["block_id"])

    source_hashes = {
        row["standard_id"]: row.get("source_sha256")
        for row in manifest["standards"]
    }
    chunks: list[dict] = []
    for paragraph in paragraphs:
        paragraph_id = paragraph["paragraph_id"]
        chunks.extend(
            _paragraph_chunks(
                paragraph,
                block_lookup,
                tables_by_paragraph.get(paragraph_id, []),
                footnotes_by_paragraph.get(paragraph_id, []),
                source_hashes.get(paragraph["standard_id"]),
                config,
            )
        )
    for table in tables:
        paragraph = paragraph_lookup.get(table.get("parent_paragraph_id"))
        chunks.extend(
            _table_chunks(
                table,
                paragraph,
                table_block_ids.get(table["table_id"], []),
                source_hashes.get(table["standard_id"]),
                config,
            )
        )

    chunks.sort(
        key=lambda row: (
            row["standard_id"],
            row["document_order"] if row["document_order"] is not None else 10**9,
            1 if row["chunk_type"] == "table" else 0,
            row["chunk_id"],
        )
    )
    report = validate_chunks(chunks, paragraphs, blocks, tables, footnotes, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    (output_dir / "chunk_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "config": asdict(config),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "CHUNK_QUALITY_REPORT.md").write_text(
        _quality_report(report, config),
        encoding="utf-8",
        newline="\n",
    )
    return report
