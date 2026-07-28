from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.hwpx_parser import (  # noqa: E402
    HwpxParser,
    map_pdf_pages,
    select_paragraphs,
)


DEFAULT_SAMPLE_NUMBERS = [
    "3.2.6",
    "4.1.2",
    "5.2.3",
    "5.5.17",
    "B3.2.17",
    "B4.1.13",
    "B5.5.17",
    "IE7",
    "IE54",
    "IE70",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_validation_markdown(sample: list[dict], hwpx: Path, pdf: Path) -> str:
    table_count = sum(len(paragraph["tables"]) for paragraph in sample)
    footnote_count = sum(len(paragraph["footnotes"]) for paragraph in sample)
    subitem_count = sum(len(paragraph["subitems"]) for paragraph in sample)
    mapped_count = sum(paragraph["pdf_page_start"] is not None for paragraph in sample)

    lines = [
        "# K-IFRS 1109 파서 시제품 검수 리포트",
        "",
        "## 실행 요약",
        "",
        f"- HWPX: `{hwpx.name}`",
        f"- PDF: `{pdf.name}`",
        f"- 대표 문단: {len(sample)}개",
        f"- 하위 항목: {subitem_count}개",
        f"- 표: {table_count}개",
        f"- 각주: {footnote_count}개",
        f"- PDF 페이지 자동 매핑: {mapped_count}/{len(sample)}개",
        "",
        "## 사람이 확인할 항목",
        "",
        "- [ ] 문단번호와 본문이 원문과 같은가",
        "- [ ] ⑴·㈎ 등의 하위 항목 순서가 같은가",
        "- [ ] 표의 행·열 및 병합 셀이 유지되었는가",
        "- [ ] 각주가 올바른 문단에 연결되었는가",
        "- [ ] 표시된 PDF 페이지에서 원문을 확인할 수 있는가",
        "",
        "## 대표 문단",
        "",
    ]

    for paragraph in sample:
        page_start = paragraph["pdf_page_start"]
        page_end = paragraph["pdf_page_end"]
        pages = "미매핑" if page_start is None else str(page_start)
        if page_start is not None and page_end != page_start:
            pages = f"{page_start}~{page_end}"
        lines.extend(
            [
                f"### {paragraph['paragraph_id']}",
                "",
                f"- 영역: `{paragraph['zone']}`",
                f"- 제목 경로: `{' > '.join(paragraph['section_path'])}`",
                f"- PDF 페이지: `{pages}`",
                f"- 페이지 매핑 신뢰도: `{paragraph['page_match_confidence']}`",
                f"- 하위 항목: `{len(paragraph['subitems'])}`",
                f"- 표: `{len(paragraph['tables'])}`",
                f"- 각주: `{len(paragraph['footnotes'])}`",
                f"- 명시적 참조: `{len(paragraph['references'])}`",
                "",
                f"> {paragraph['text'][:280]}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the K-IFRS 1109 parser prototype.")
    parser.add_argument(
        "--hwpx",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "standards" / "K-IFRS_1109.hwpx",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "standards" / "K-IFRS_1109.pdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation" / "parser_prototype_1109",
    )
    args = parser.parse_args()

    if not args.hwpx.exists() or not args.pdf.exists():
        parser.error("HWPX와 PDF 원본 경로를 확인하세요.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed = HwpxParser("1109").parse(args.hwpx)
    sample = select_paragraphs(parsed, DEFAULT_SAMPLE_NUMBERS)
    map_pdf_pages(sample, args.pdf)

    manifest = {
        "hwpx": {"path": str(args.hwpx), "sha256": sha256(args.hwpx)},
        "pdf": {"path": str(args.pdf), "sha256": sha256(args.pdf)},
        "sample_numbers": DEFAULT_SAMPLE_NUMBERS,
        "found_numbers": [paragraph["number"] for paragraph in sample],
    }

    (args.output_dir / "sample_paragraphs.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "VALIDATION_REPORT.md").write_text(
        build_validation_markdown(sample, args.hwpx, args.pdf), encoding="utf-8"
    )

    print(json.dumps({
        "parsed_paragraphs": len(parsed["paragraphs"]),
        "sample_paragraphs": len(sample),
        "sample_numbers": [paragraph["number"] for paragraph in sample],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
