import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.hwpx_parser import search_normalize
from accounting_rag.ingestion.pdf_page_mapper import (
    PageMappingConfig,
    match_page_range,
    validate_page_mapping,
)


class PdfPageMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PageMappingConfig(anchor_chars=8, minimum_text_chars=4)
        self.pages = [
            search_normalize("첫 페이지의 일반적인 설명입니다."),
            search_normalize("금융자산의 계약상 현금흐름을 평가한다."),
            search_normalize("평가 결과와 예외 조건을 다음 페이지에서 설명한다."),
        ]

    def test_maps_exact_text_to_page(self) -> None:
        result = match_page_range(
            "금융자산의 계약상 현금흐름을 평가한다.", self.pages, self.config
        )
        self.assertEqual(result["pdf_page_start"], 2)
        self.assertEqual(result["pdf_page_end"], 2)
        self.assertEqual(result["page_match_method"], "exact_anchor")

    def test_maps_text_spanning_adjacent_pages(self) -> None:
        result = match_page_range(
            "금융자산의 계약상 현금흐름을 평가한다. 평가 결과와 예외 조건을 다음 페이지에서 설명한다.",
            self.pages,
            self.config,
        )
        self.assertEqual((result["pdf_page_start"], result["pdf_page_end"]), (2, 3))

    def test_short_unsearchable_text_remains_unresolved(self) -> None:
        result = match_page_range("가", self.pages, self.config)
        self.assertIsNone(result["pdf_page_start"])

    def test_validation_rejects_out_of_range_page(self) -> None:
        paragraph = {
            "paragraph_id": "P1",
            "standard_id": "1109",
            "zone": "standard_body",
            "pdf_page_start": 4,
            "pdf_page_end": 4,
            "page_match_confidence": 1.0,
            "page_match_method": "exact_anchor",
        }
        pages = [{"standard_id": "1109", "pdf_page": i} for i in range(1, 4)]
        with self.assertRaises(ValueError):
            validate_page_mapping([paragraph], [], [], [], pages, self.config)


if __name__ == "__main__":
    unittest.main()
