from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from math import ceil
from pathlib import Path
from typing import Iterable

from .hwpx_parser import search_normalize


MAPPING_SCHEMA_VERSION = "1.0"
SOURCE_FILES = (
    "paragraphs.jsonl",
    "blocks.jsonl",
    "tables.jsonl",
    "footnotes.jsonl",
)


@dataclass(frozen=True)
class PageMappingConfig:
    anchor_chars: int = 32
    minimum_text_chars: int = 8
    fuzzy_threshold: float = 0.82
    fuzzy_window_pages: int = 8
    maximum_span_pages: int = 12
    child_window_pages: int = 3
    review_confidence_below: float = 0.75

    @classmethod
    def from_json(cls, path: Path) -> "PageMappingConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _anchors(text: str, size: int) -> list[str]:
    normalized = search_normalize(text)
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]
    last = len(normalized) - size
    positions = (0, last // 3, (2 * last) // 3, last)
    return list(dict.fromkeys(normalized[position : position + size] for position in positions))


def match_page_range(
    text: str,
    pages: list[str],
    config: PageMappingConfig,
    hint_page: int | None = None,
    allowed_pages: range | None = None,
) -> dict:
    normalized = search_normalize(text)
    empty = {
        "pdf_page_start": None,
        "pdf_page_end": None,
        "page_match_confidence": 0.0,
        "page_match_method": "unresolved",
        "page_match_anchor_count": 0,
        "page_match_anchor_hits": 0,
        "page_match_ambiguous": False,
        "page_match_candidate_pages": [],
        "page_match_anchor_pages": [],
    }
    if len(normalized) < config.minimum_text_chars:
        return empty

    anchors = _anchors(text, config.anchor_chars)
    candidates = list(allowed_pages or range(1, len(pages) + 1))
    occurrences: dict[str, list[int]] = {
        anchor: [page for page in candidates if anchor in pages[page - 1]]
        for anchor in anchors
    }
    scores = Counter(page for matches in occurrences.values() for page in matches)
    method = "exact_anchor"
    if scores:
        best_score = max(scores.values())
        best_pages = [page for page, score in scores.items() if score == best_score]
        seed = min(
            best_pages,
            key=lambda page: (abs(page - hint_page), page)
            if hint_page is not None
            else (page,),
        )
        ambiguity = len(best_pages) > 1
        radius = min(
            config.maximum_span_pages,
            max(2, ceil(len(normalized) / 1200) + 1),
        )
        matched_pages = []
        previous_match = max(1, seed - radius)
        for found in occurrences.values():
            valid = [
                page
                for page in found
                if abs(page - seed) <= radius and page >= previous_match
            ]
            if not valid:
                continue
            selected = min(valid, key=lambda page: (abs(page - seed), page))
            matched_pages.append(selected)
            previous_match = selected
        hits = len(matched_pages)
        confidence = hits / len(anchors)
        if ambiguity:
            confidence *= 0.75 if hint_page is not None else 0.5
        return {
            "pdf_page_start": min(matched_pages),
            "pdf_page_end": max(matched_pages),
            "page_match_confidence": round(confidence, 3),
            "page_match_method": method,
            "page_match_anchor_count": len(anchors),
            "page_match_anchor_hits": hits,
            "page_match_ambiguous": ambiguity,
            "page_match_candidate_pages": sorted(best_pages),
            "page_match_anchor_pages": [
                {"anchor_index": index, "pages": found}
                for index, found in enumerate(occurrences.values(), start=1)
                if found
            ],
        }

    if hint_page is None:
        return empty
    window = range(
        max(1, hint_page - config.fuzzy_window_pages),
        min(len(pages), hint_page + config.fuzzy_window_pages) + 1,
    )
    probe = normalized[: config.anchor_chars]
    ratios = []
    for page in window:
        longest = SequenceMatcher(None, probe, pages[page - 1], autojunk=False).find_longest_match()
        ratios.append((longest.size / len(probe), page))
    ratio, page = max(ratios, default=(0.0, hint_page))
    if ratio < config.fuzzy_threshold:
        return empty
    return {
        "pdf_page_start": page,
        "pdf_page_end": page,
        "page_match_confidence": round(ratio * 0.75, 3),
        "page_match_method": "fuzzy_anchor",
        "page_match_anchor_count": len(anchors),
        "page_match_anchor_hits": 1,
        "page_match_ambiguous": False,
        "page_match_candidate_pages": [page],
        "page_match_anchor_pages": [],
    }


def _inherited(parent: dict | None) -> dict:
    if not parent or parent.get("pdf_page_start") is None:
        return {
            "pdf_page_start": None,
            "pdf_page_end": None,
            "page_match_confidence": 0.0,
            "page_match_method": "unresolved",
            "page_match_anchor_count": 0,
            "page_match_anchor_hits": 0,
            "page_match_ambiguous": False,
        }
    return {
        "pdf_page_start": parent["pdf_page_start"],
        "pdf_page_end": parent["pdf_page_end"],
        "page_match_confidence": round(parent["page_match_confidence"] * 0.8, 3),
        "page_match_method": "inherited_parent",
        "page_match_anchor_count": 0,
        "page_match_anchor_hits": 0,
        "page_match_ambiguous": parent.get("page_match_ambiguous", False),
        "page_match_candidate_pages": parent.get("page_match_candidate_pages", []),
        "page_match_anchor_pages": [],
    }


def _table_text(table: dict) -> str:
    cells = sorted(table["cells"], key=lambda cell: (cell["row"], cell["column"]))
    return " ".join(cell["text"] for cell in cells if cell.get("text"))


def _extract_pdf_pages(pdf_path: Path) -> tuple[list[dict], list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    labels = reader.page_labels
    records = []
    normalized = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        records.append(
            {
                "standard_id": pdf_path.stem.rsplit("_", 1)[-1],
                "pdf_page": index,
                "pdf_page_label": labels[index - 1] if index <= len(labels) else str(index),
                "text": text,
                "char_count": len(text),
            }
        )
        normalized.append(search_normalize(text))
    return records, normalized


def _map_children(
    rows: list[dict],
    text_getter,
    pages_by_standard: dict[str, list[str]],
    paragraph_lookup: dict[str, dict],
    config: PageMappingConfig,
) -> None:
    for row in rows:
        parent = paragraph_lookup.get(row.get("parent_paragraph_id"))
        allowed = None
        hint = None
        if parent and parent.get("pdf_page_start") is not None:
            hint = parent["pdf_page_start"]
            allowed = range(
                parent["pdf_page_start"],
                min(
                    len(pages_by_standard[row["standard_id"]]),
                    parent["pdf_page_end"] + config.child_window_pages,
                )
                + 1,
            )
        match = match_page_range(
            text_getter(row),
            pages_by_standard[row["standard_id"]],
            config,
            hint_page=hint,
            allowed_pages=allowed,
        )
        if match["pdf_page_start"] is None:
            match = _inherited(parent)
        row.update(match)


def _refine_paragraph_ranges(
    paragraphs: list[dict],
    blocks: list[dict],
    tables: list[dict],
    footnotes: list[dict],
) -> None:
    direct_children: dict[str, list[dict]] = defaultdict(list)
    for row in [*blocks, *tables, *footnotes]:
        parent_id = row.get("parent_paragraph_id")
        if (
            parent_id
            and row.get("pdf_page_start") is not None
            and row.get("page_match_method") in {"exact_anchor", "fuzzy_anchor"}
        ):
            direct_children[parent_id].append(row)

    for paragraph in paragraphs:
        sources = list(direct_children.get(paragraph["paragraph_id"], []))
        if paragraph.get("pdf_page_start") is not None:
            sources.append(paragraph)
        if not sources:
            continue
        was_unresolved = paragraph.get("pdf_page_start") is None
        paragraph["pdf_page_start"] = min(row["pdf_page_start"] for row in sources)
        paragraph["pdf_page_end"] = max(row["pdf_page_end"] for row in sources)
        paragraph["page_match_confidence"] = round(
            min(row["page_match_confidence"] for row in sources), 3
        )
        if was_unresolved:
            paragraph["page_match_method"] = "derived_from_children"



def _infer_unmapped_paragraphs(paragraphs: list[dict]) -> None:
    by_standard: dict[str, list[dict]] = defaultdict(list)
    for paragraph in paragraphs:
        by_standard[paragraph["standard_id"]].append(paragraph)
    for standard_paragraphs in by_standard.values():
        standard_paragraphs.sort(key=lambda row: row["document_order"])
        for index, paragraph in enumerate(standard_paragraphs):
            if paragraph.get("pdf_page_start") is not None:
                continue
            next_mapped = next(
                (
                    row
                    for row in standard_paragraphs[index + 1 :]
                    if row.get("pdf_page_start") is not None
                ),
                None,
            )
            if next_mapped is None:
                continue
            paragraph.update(
                {
                    "pdf_page_start": next_mapped["pdf_page_start"],
                    "pdf_page_end": next_mapped["pdf_page_start"],
                    "page_match_confidence": 0.5,
                    "page_match_method": "inferred_next_paragraph",
                    "page_match_anchor_count": 0,
                    "page_match_anchor_hits": 0,
                    "page_match_ambiguous": False,
                    "page_match_candidate_pages": [next_mapped["pdf_page_start"]],
                    "page_match_anchor_pages": [],
                }
            )


def _infer_substantive_orphan_tables(tables: list[dict]) -> None:
    by_standard: dict[str, list[dict]] = defaultdict(list)
    for table in tables:
        by_standard[table["standard_id"]].append(table)
    for standard_tables in by_standard.values():
        standard_tables.sort(key=lambda row: int(row["table_id"].rsplit("-", 1)[-1]))
        for index, table in enumerate(standard_tables):
            substantive = (
                table.get("rows", 0) > 1
                or table.get("columns", 0) > 1
                or len(_table_text(table)) >= 40
            )
            if table.get("pdf_page_start") is not None or not substantive:
                continue
            previous = next(
                (
                    row
                    for row in reversed(standard_tables[:index])
                    if row.get("pdf_page_end") is not None
                ),
                None,
            )
            following = next(
                (
                    row
                    for row in standard_tables[index + 1 :]
                    if row.get("pdf_page_start") is not None
                ),
                None,
            )
            if (
                previous is None
                or following is None
                or following["pdf_page_start"] - previous["pdf_page_end"] > 2
            ):
                continue
            table.update(
                {
                    "pdf_page_start": previous["pdf_page_end"],
                    "pdf_page_end": following["pdf_page_start"],
                    "page_match_confidence": 0.5,
                    "page_match_method": "inferred_adjacent_table",
                    "page_match_anchor_count": 0,
                    "page_match_anchor_hits": 0,
                    "page_match_ambiguous": False,
                    "page_match_candidate_pages": [
                        previous["pdf_page_end"],
                        following["pdf_page_start"],
                    ],
                    "page_match_anchor_pages": [],
                }
            )


def validate_page_mapping(
    paragraphs: list[dict],
    blocks: list[dict],
    tables: list[dict],
    footnotes: list[dict],
    page_records: list[dict],
    config: PageMappingConfig,
) -> dict:
    page_counts = Counter(row["standard_id"] for row in page_records)
    all_rows = {
        "paragraphs": paragraphs,
        "blocks": blocks,
        "tables": tables,
        "footnotes": footnotes,
    }
    for label, rows in all_rows.items():
        for row in rows:
            start, end = row.get("pdf_page_start"), row.get("pdf_page_end")
            if (start is None) != (end is None):
                raise ValueError(f"Partial page range in {label}: {row}")
            if start is not None and not (1 <= start <= end <= page_counts[row["standard_id"]]):
                raise ValueError(f"Invalid page range in {label}: {row}")

    searchable_paragraphs = [row for row in paragraphs if row["zone"] not in {"front_matter", "committee_resolution", "amendment_history", "ifrs_comparison"}]
    mapped_searchable = sum(row["pdf_page_start"] is not None for row in searchable_paragraphs)
    report = {
        "valid": True,
        "pdf_pages": len(page_records),
        "page_counts": dict(sorted(page_counts.items())),
        "paragraphs": len(paragraphs),
        "mapped_paragraphs": sum(row["pdf_page_start"] is not None for row in paragraphs),
        "searchable_paragraphs": len(searchable_paragraphs),
        "mapped_searchable_paragraphs": mapped_searchable,
        "unmapped_searchable_paragraphs": len(searchable_paragraphs) - mapped_searchable,
        "low_confidence_paragraphs": sum(
            row["pdf_page_start"] is not None
            and row["page_match_confidence"] < config.review_confidence_below
            for row in searchable_paragraphs
        ),
        "ambiguous_paragraphs": sum(row.get("page_match_ambiguous", False) for row in searchable_paragraphs),
        "mapped_blocks": sum(row["pdf_page_start"] is not None for row in blocks),
        "mapped_tables": sum(row["pdf_page_start"] is not None for row in tables),
        "mapped_footnotes": sum(row["pdf_page_start"] is not None for row in footnotes),
        "paragraph_methods": dict(Counter(row["page_match_method"] for row in paragraphs)),
    }
    return report


def _quality_markdown(report: dict, config: PageMappingConfig) -> str:
    lines = [
        "# PDF 페이지 매핑 품질보고서",
        "",
        "## 결과",
        "",
        f"- PDF 페이지: {report['pdf_pages']:,}쪽",
        f"- 전체 문단 매핑: {report['mapped_paragraphs']:,}/{report['paragraphs']:,}개",
        f"- 검색 대상 문단 매핑: {report['mapped_searchable_paragraphs']:,}/{report['searchable_paragraphs']:,}개",
        f"- 검색 대상 미매핑: {report['unmapped_searchable_paragraphs']:,}개",
        f"- 신뢰도 {config.review_confidence_below:.2f} 미만 검토 대상: {report['low_confidence_paragraphs']:,}개",
        f"- 복수 후보 검토 대상: {report['ambiguous_paragraphs']:,}개",
        f"- 원문 Block 매핑: {report['mapped_blocks']:,}개",
        f"- 표 매핑: {report['mapped_tables']:,}개",
        f"- 각주 매핑: {report['mapped_footnotes']:,}개",
        "",
        "## 기준서별 PDF 페이지",
        "",
    ]
    lines.extend(f"- 제{standard_id}호: {count:,}쪽" for standard_id, count in report["page_counts"].items())
    lines.extend(["", "## 매핑 방식", ""])
    lines.extend(f"- `{method}`: {count:,}개 문단" for method, count in sorted(report["paragraph_methods"].items()))
    lines.extend(
        [
            "",
            "## 판정 원칙",
            "",
            "- HWPX가 구조의 기준이고 PDF는 페이지 위치의 기준입니다.",
            "- 문단의 앞·중간·끝 문자열이 실제 PDF 텍스트에 나타나는 페이지를 찾습니다.",
            "- 문서 순서는 동일 문구가 여러 페이지에 있을 때 가장 가까운 후보를 고르는 보조 정보로만 사용합니다.",
            "- 표·각주·하위 Block은 부모 문단 주변에서 먼저 찾고, 직접 찾지 못하면 부모 범위를 낮은 신뢰도로 상속합니다.",
            "- 낮은 신뢰도와 복수 후보는 자동 확정 결과와 함께 검토 목록에 보존합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def build_page_mapping(
    processed_dir: Path,
    pdf_dir: Path,
    config: PageMappingConfig | None = None,
) -> dict:
    config = config or PageMappingConfig()
    paragraphs = read_jsonl(processed_dir / "paragraphs.jsonl")
    blocks = read_jsonl(processed_dir / "blocks.jsonl")
    tables = read_jsonl(processed_dir / "tables.jsonl")
    footnotes = read_jsonl(processed_dir / "footnotes.jsonl")

    page_records: list[dict] = []
    pages_by_standard: dict[str, list[str]] = {}
    pdf_manifest = []
    for standard_id in ("1032", "1039", "1107", "1109"):
        pdf_path = pdf_dir / f"K-IFRS_{standard_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        records, normalized_pages = _extract_pdf_pages(pdf_path)
        page_records.extend(records)
        pages_by_standard[standard_id] = normalized_pages
        pdf_manifest.append(
            {
                "standard_id": standard_id,
                "source_file": pdf_path.name,
                "source_sha256": _sha256(pdf_path),
                "pages": len(records),
            }
        )

    blocks_by_paragraph: dict[str, list[dict]] = defaultdict(list)
    for block in blocks:
        if block.get("parent_paragraph_id"):
            blocks_by_paragraph[block["parent_paragraph_id"]].append(block)

    previous_page: dict[str, int] = {}
    for paragraph in paragraphs:
        standard_id = paragraph["standard_id"]
        content = " ".join(
            block["text"]
            for block in blocks_by_paragraph.get(paragraph["paragraph_id"], [])
            if block.get("text")
        ) or paragraph["text"]
        match = match_page_range(
            content,
            pages_by_standard[standard_id],
            config,
            hint_page=previous_page.get(standard_id),
        )
        paragraph.update(match)
        if match["pdf_page_end"] is not None and match["page_match_confidence"] >= config.review_confidence_below:
            previous_page[standard_id] = match["pdf_page_end"]

    _infer_unmapped_paragraphs(paragraphs)
    paragraph_lookup = {row["paragraph_id"]: row for row in paragraphs}
    _map_children(blocks, lambda row: row["text"], pages_by_standard, paragraph_lookup, config)
    _map_children(tables, _table_text, pages_by_standard, paragraph_lookup, config)
    _map_children(footnotes, lambda row: row["text"], pages_by_standard, paragraph_lookup, config)
    _infer_substantive_orphan_tables(tables)
    _refine_paragraph_ranges(paragraphs, blocks, tables, footnotes)

    report = validate_page_mapping(paragraphs, blocks, tables, footnotes, page_records, config)
    review_rows = [
        {
            "paragraph_id": row["paragraph_id"],
            "standard_id": row["standard_id"],
            "number": row["number"],
            "zone": row["zone"],
            "pdf_page_start": row["pdf_page_start"],
            "pdf_page_end": row["pdf_page_end"],
            "page_match_confidence": row["page_match_confidence"],
            "page_match_method": row["page_match_method"],
            "page_match_ambiguous": row["page_match_ambiguous"],
            "page_match_candidate_pages": row.get("page_match_candidate_pages", []),
            "page_match_anchor_pages": row.get("page_match_anchor_pages", []),
            "text_preview": row["text"][:240],
        }
        for row in paragraphs
        if row["pdf_page_start"] is None
        or row["page_match_confidence"] < config.review_confidence_below
        or row["page_match_ambiguous"]
    ]

    write_jsonl(processed_dir / "paragraphs.jsonl", paragraphs)
    write_jsonl(processed_dir / "blocks.jsonl", blocks)
    write_jsonl(processed_dir / "tables.jsonl", tables)
    write_jsonl(processed_dir / "footnotes.jsonl", footnotes)
    write_jsonl(processed_dir / "pdf_pages.jsonl", page_records)
    write_jsonl(processed_dir / "page_mapping_review.jsonl", review_rows)
    (processed_dir / "page_mapping_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "config": asdict(config),
                "pdfs": pdf_manifest,
                **report,
                "review_items": len(review_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (processed_dir / "PAGE_MAPPING_QUALITY_REPORT.md").write_text(
        _quality_markdown(report, config), encoding="utf-8", newline="\n"
    )
    return report
