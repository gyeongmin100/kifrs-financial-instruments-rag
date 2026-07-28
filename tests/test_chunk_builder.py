import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.chunk_builder import (
    ChunkingConfig,
    _paragraph_chunks,
    _table_chunks,
    build_chunks,
)


def paragraph(**overrides):
    row = {
        "paragraph_id": "KIFRS1109-1",
        "number": "1",
        "zone": "standard_body",
        "section_path": ["금융상품"],
        "text": "본문",
        "source_section": "Contents/section0.xml",
        "xml_index": 1,
        "document_order": 1,
        "occurrence": 1,
        "block_ids": ["B1"],
        "subitems": [],
        "pdf_page_start": 10,
        "pdf_page_end": 10,
        "page_match_confidence": 1.0,
        "standard_id": "1109",
    }
    row.update(overrides)
    return row


def block(block_id, text, block_type="paragraph", table_ids=None):
    return {
        "standard_id": "1109",
        "block_id": block_id,
        "block_type": block_type,
        "zone": "standard_body",
        "text": text,
        "source_section": "Contents/section0.xml",
        "xml_index": 1,
        "document_order": 1,
        "searchable": True,
        "search_priority": 3,
        "section_path": ["금융상품"],
        "parent_paragraph_id": "KIFRS1109-1",
        "table_ids": table_ids or [],
        "footnote_ids": [],
        "references": [],
    }


class ChunkBuilderTests(unittest.TestCase):
    def test_keeps_short_paragraph_and_subitems_together(self) -> None:
        source = paragraph(
            block_ids=["B1", "B2", "B3"],
            subitems=[
                {"marker": "⑴", "text": "첫째 조건", "xml_index": 2},
                {"marker": "⑵", "text": "둘째 조건", "xml_index": 3},
            ],
        )
        blocks = {
            "B1": block("B1", "1 본문"),
            "B2": block("B2", "⑴ 첫째 조건", "subitem"),
            "B3": block("B3", "⑵ 둘째 조건", "subitem"),
        }

        chunks = _paragraph_chunks(
            source, blocks, [], [], "hash", ChunkingConfig()
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0]["source_subparagraph_ids"],
            ["KIFRS1109-1-S01", "KIFRS1109-1-S02"],
        )
        self.assertIn("⑵ 둘째 조건", chunks[0]["text"])

    def test_splits_long_paragraph_at_subitem_boundaries(self) -> None:
        source = paragraph(
            block_ids=["B1", "B2", "B3"],
            subitems=[
                {"marker": "⑴", "text": "가" * 40, "xml_index": 2},
                {"marker": "⑵", "text": "나" * 40, "xml_index": 3},
            ],
        )
        blocks = {
            "B1": block("B1", "1 도입"),
            "B2": block("B2", "⑴ " + "가" * 40, "subitem"),
            "B3": block("B3", "⑵ " + "나" * 40, "subitem"),
        }
        config = ChunkingConfig(paragraph_target_chars=50, paragraph_max_chars=70)

        chunks = _paragraph_chunks(source, blocks, [], [], "hash", config)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["char_count"] <= 70 for chunk in chunks))
        self.assertEqual(chunks[0]["part_count"], len(chunks))
        self.assertIn("문단 도입:", chunks[-1]["contextualized_text"])

    def test_deleted_paragraph_is_preserved_but_not_searchable(self) -> None:
        source = paragraph(text="[국제회계기준위원회에서 삭제함]")
        blocks = {"B1": block("B1", "1 [국제회계기준위원회에서 삭제함]")}

        chunk = _paragraph_chunks(
            source, blocks, [], [], "hash", ChunkingConfig()
        )[0]

        self.assertTrue(chunk["inactive"])
        self.assertFalse(chunk["searchable"])

    def test_continuation_block_is_included_and_traceable(self) -> None:
        source = paragraph(block_ids=["B1", "B2"])
        blocks = {
            "B1": block("B1", "1 본문"),
            "B2": block("B2", "문단에 이어지는 추가 설명", "continuation"),
        }
        blocks["B1"].update(
            pdf_page_start=10, pdf_page_end=10, page_match_confidence=1.0
        )
        blocks["B2"].update(
            pdf_page_start=11, pdf_page_end=11, page_match_confidence=0.75
        )

        chunk = _paragraph_chunks(
            source, blocks, [], [], "hash", ChunkingConfig()
        )[0]

        self.assertIn("추가 설명", chunk["text"])
        self.assertEqual(chunk["block_ids"], ["B1", "B2"])
        self.assertEqual((chunk["pdf_page_start"], chunk["pdf_page_end"]), (10, 11))
        self.assertEqual(chunk["page_match_confidence"], 0.75)

    def test_long_table_splits_by_rows_and_repeats_header(self) -> None:
        table = {
            "standard_id": "1109",
            "zone": "standard_body",
            "section_path": ["금융상품"],
            "table_id": "T1",
            "parent_paragraph_id": "KIFRS1109-1",
            "rows": 3,
            "columns": 1,
            "repeat_header": True,
            "pdf_page_start": 12,
            "pdf_page_end": 13,
            "page_match_confidence": 0.75,
            "cells": [
                {"row": 0, "column": 0, "row_span": 1, "column_span": 1, "text": "제목"},
                {"row": 1, "column": 0, "row_span": 1, "column_span": 1, "text": "가" * 50},
                {"row": 2, "column": 0, "row_span": 1, "column_span": 1, "text": "나" * 50},
            ],
        }
        config = ChunkingConfig(table_target_chars=70, table_max_chars=100)

        chunks = _table_chunks(table, paragraph(), ["B1"], "hash", config)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["char_count"] <= 100 for chunk in chunks))
        self.assertTrue(all("제목" in chunk["text"] for chunk in chunks))
        self.assertTrue(all(chunk["pdf_page_start"] == 12 for chunk in chunks))
        self.assertTrue(all(chunk["pdf_page_end"] == 13 for chunk in chunks))

    def test_build_chunks_writes_valid_traceable_dataset(self) -> None:
        source_paragraph = paragraph()
        source_block = block("B1", "1 본문")
        manifest = {
            "standards": [{"standard_id": "1109", "source_sha256": "hash"}],
            "totals": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            output = root / "chunks"
            processed.mkdir()
            for name, rows in {
                "paragraphs.jsonl": [source_paragraph],
                "blocks.jsonl": [source_block],
                "tables.jsonl": [],
                "footnotes.jsonl": [],
            }.items():
                (processed / name).write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )
            (processed / "document_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            report = build_chunks(processed, output)

            self.assertTrue(report["valid"])
            self.assertTrue((output / "chunks.jsonl").exists())
            self.assertTrue((output / "CHUNK_QUALITY_REPORT.md").exists())


if __name__ == "__main__":
    unittest.main()
