import unittest
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.ingestion.hwpx_parser import (
    PARAGRAPH_NUMBER_RE,
    HwpxParser,
    _mixed_text,
    _parse_table,
    parse_references,
)


NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"


class HwpxParserTests(unittest.TestCase):
    def test_section_title_with_parenthesized_section_number_is_heading(self) -> None:
        from accounting_rag.ingestion.hwpx_parser import _is_heading

        self.assertTrue(_is_heading("금융부채의 제거(제3.3절)", None))

    def test_revision_text_uses_inserted_and_excludes_deleted(self) -> None:
        text = ET.fromstring(
            f"""<hp:t xmlns:hp="{NS}">앞
            <hp:deleteBegin/>삭제 문구<hp:deleteEnd/>
            <hp:insertBegin/>현재 문구<hp:insertEnd/>뒤</hp:t>"""
        )

        result = " ".join(_mixed_text(text).split())

        self.assertNotIn("삭제 문구", result)
        self.assertIn("현재 문구", result)
        self.assertTrue(result.startswith("앞"))
        self.assertTrue(result.endswith("뒤"))

    def test_explicit_cross_standard_and_local_paragraph_references(self) -> None:
        references = parse_references(
            "기업회계기준서 제1039호 문단 89~94를 적용한다. 문단 7.2.21도 적용한다.",
            source_standard="1109",
        )

        self.assertTrue(any(reference.target_standard == "1039" for reference in references))
        self.assertTrue(any(
            reference.target_standard == "1039"
            and reference.target_paragraph_start == "89"
            and reference.target_paragraph_end == "94"
            for reference in references
        ))
        self.assertTrue(any(
            reference.target_standard == "1109"
            and reference.target_paragraph_start == "7.2.21"
            for reference in references
        ))

    def test_real_reference_variants_and_ordinary_range(self) -> None:
        references = parse_references(
            "기업회계기준서 제1039호 문단89⑵와 101⑵, 문단 BC220E–BC220G를 "
            "적용한다. 문단 BA.2와 BCE.174도 본다. 손실률은 10~20%이다.",
            source_standard="1109",
        )

        paragraph_references = [
            reference for reference in references if reference.kind != "standard"
        ]
        self.assertTrue(any(
            reference.target_standard == "1039"
            and reference.target_paragraph_start == "89"
            and reference.target_subitem_start == "⑵"
            for reference in paragraph_references
        ))
        self.assertTrue(any(
            reference.target_paragraph_start == "BC220E"
            and reference.target_paragraph_end == "BC220G"
            and reference.range_delimiter == "–"
            for reference in paragraph_references
        ))
        self.assertTrue(any(
            reference.target_standard == "1109"
            and reference.target_paragraph_start == "BA.2"
            for reference in paragraph_references
        ))
        self.assertFalse(any(
            reference.target_paragraph_start == "10"
            for reference in paragraph_references
        ))

    def test_subitem_range_and_ias_alias(self) -> None:
        references = parse_references(
            "IAS 39 문단 42E(1)~(4)를 참고한다.", source_standard="1109"
        )

        reference = next(
            item for item in references if item.kind == "subitem_range"
        )
        self.assertEqual(reference.target_standard, "1039")
        self.assertEqual(reference.target_paragraph_start, "42E")
        self.assertEqual(reference.target_subitem_start, "(1)")
        self.assertEqual(reference.target_subitem_end, "(4)")

    def test_standard_context_distinguishes_actor_from_reference_target(self) -> None:
        references = parse_references(
            "기업회계기준서 제1109호에 따라 문단 3과 4를 개정한다. "
            "기업회계기준서 제1109호에 따라 회계처리하고 그 기준서 문단 B2.7을 적용한다.",
            source_standard="1032",
        )
        paragraph_references = [
            reference for reference in references if reference.target_paragraph_start
        ]

        self.assertEqual(paragraph_references[0].target_standard, "1032")
        self.assertEqual(paragraph_references[1].target_standard, "1032")
        self.assertEqual(paragraph_references[2].target_standard, "1109")

    def test_standard_title_and_fullwidth_ifrs_number_are_direct_context(self) -> None:
        references = parse_references(
            "기업회계기준서 제1039호 ‘금융상품: 인식과 측정’ 문단 89~94와 "
            "IFRS ９ 문단 3.2.6 및 기업회계기준서 1109호 문단 4.1.2A를 참고한다.",
            source_standard="1107",
        )
        paragraph_references = [
            reference for reference in references if reference.target_paragraph_start
        ]

        self.assertEqual(paragraph_references[0].target_standard, "1039")
        self.assertEqual(paragraph_references[1].target_standard, "1109")
        self.assertEqual(paragraph_references[2].target_standard, "1109")

    def test_anaphoric_standard_survives_dots_in_prior_paragraph_number(self) -> None:
        references = parse_references(
            "기업회계기준서 제1109호 문단 2.3A를 충족하고 그 기준서 문단 B2.7~B2.8을 적용한다.",
            source_standard="1107",
        )
        range_reference = next(
            reference
            for reference in references
            if reference.target_paragraph_start == "B2.7"
        )

        self.assertEqual(range_reference.target_standard, "1109")

    def test_paragraph_number_variants(self) -> None:
        variants = ["1 본문", "35A 본문", "한97A.1 본문", "IG13C 본문", "BA.2 본문", "BCIN.1 본문", "BCE.174 본문"]

        parsed = [PARAGRAPH_NUMBER_RE.match(text).group("number") for text in variants]

        self.assertEqual(
            parsed,
            ["1", "35A", "한97A.1", "IG13C", "BA.2", "BCIN.1", "BCE.174"],
        )

    def test_section_zero_is_preserved_and_body_is_detected(self) -> None:
        xml = f"""<hs:sec xmlns:hs="urn:test" xmlns:hp="{NS}">
          <hp:p><hp:run><hp:t>저작권</hp:t></hp:run></hp:p>
          <hp:p><hp:run><hp:t>기업회계기준서 제1032호</hp:t></hp:run></hp:p>
          <hp:p><hp:run><hp:t>1 본문 내용</hp:t></hp:run></hp:p>
        </hs:sec>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.hwpx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("Contents/section0.xml", xml)

            document = HwpxParser("1032").parse(path)

        self.assertEqual(
            [paragraph["number"] for paragraph in document["paragraphs"]], ["1"]
        )
        self.assertEqual(document["blocks"][0]["zone"], "front_matter")
        self.assertEqual(document["blocks"][2]["zone"], "standard_body")

    def test_table_preserves_addresses_and_spans(self) -> None:
        table = ET.fromstring(
            f"""<hp:tbl xmlns:hp="{NS}" rowCnt="1" colCnt="2" repeatHeader="1">
              <hp:tr>
                <hp:tc><hp:subList><hp:p><hp:run><hp:t>항목</hp:t></hp:run></hp:p></hp:subList>
                  <hp:cellAddr rowAddr="0" colAddr="0"/><hp:cellSpan rowSpan="1" colSpan="1"/></hp:tc>
                <hp:tc><hp:subList><hp:p><hp:run><hp:t>분석</hp:t></hp:run></hp:p></hp:subList>
                  <hp:cellAddr rowAddr="0" colAddr="1"/><hp:cellSpan rowSpan="1" colSpan="1"/></hp:tc>
              </hp:tr>
            </hp:tbl>"""
        )

        parsed = _parse_table(table, "T-1")

        self.assertEqual(parsed.rows, 1)
        self.assertEqual(parsed.columns, 2)
        self.assertTrue(parsed.repeat_header)
        self.assertEqual([cell.text for cell in parsed.cells], ["항목", "분석"])
        self.assertEqual(parsed.cells[1].column, 1)


if __name__ == "__main__":
    unittest.main()
