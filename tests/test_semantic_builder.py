import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.graph.semantic_builder import (
    build_semantic_kg, concept_id, concept_key, extract_official_definitions, normalize_text,
)


class SemanticBuilderTests(unittest.TestCase):
    def test_concept_id_is_stable_across_spacing(self):
        self.assertEqual(concept_id("기대 신용손실"), concept_id("기대   신용손실"))
        self.assertEqual(concept_key(" 풋가능 금융상품 "), "풋가능금융상품")

    def test_table_definitions_require_term_and_definition_columns(self):
        tables = [{
            "table_id": "T1", "standard_id": "1109", "zone": "appendix_definitions",
            "cells": [
                {"row": 0, "column": 0, "text": "기대신용손실"},
                {"row": 0, "column": 1, "text": "현금부족액의 확률가중 추정치"},
                {"row": 1, "column": 0, "text": "정의 없는 용어"},
            ],
        }]
        definitions = extract_official_definitions(self._required_blocks(), tables)
        expected = next(row for row in definitions if row["term"] == "기대신용손실")
        self.assertEqual(expected["source_id"], "T1")
        self.assertFalse(any(row["term"] == "정의 없는 용어" for row in definitions))

    def test_1039_subitems_remain_in_parent_definition(self):
        definitions = extract_official_definitions(self._required_blocks(), [])
        target = next(row for row in definitions if row["term"] == "위험회피대상항목")
        self.assertIn("조건 하나", target["definition"])

    def test_one_definition_table_can_define_multiple_concepts(self):
        # The production 1109 table contains many term-definition rows in one Table node.
        # Each term must retain its own definition edge instead of the last row winning.
        project = Path(__file__).resolve().parents[1]
        if not (project / "data" / "processed" / "tables.jsonl").exists():
            self.skipTest("processed fixture is not available")
        graph = build_semantic_kg(project / "data" / "processed")
        edges = [row for row in graph["mentions"]
                 if row["source_id"] == "KIFRS1109-T-0014" and row["role"] == "definition"]
        self.assertGreater(len(edges), 20)

    @staticmethod
    def _required_blocks():
        rows = []
        def add(standard, order, text):
            rows.append({"block_id": f"KIFRS{standard}-BLK-{order:05d}",
                         "standard_id": standard, "document_order": order, "text": text})
        add("1032", 83, "금융상품: 계약")
        for order in range(84, 96): add("1032", order, "금융자산은 자산을 말한다")
        for order in range(96, 107): add("1032", order, "금융부채는 부채를 말한다")
        add("1032", 107, "지분상품: 잔여지분 계약")
        add("1032", 108, "공정가치: 정상거래의 가격")
        add("1032", 109, "풋가능 금융상품: 환매 요구 권리가 있는 상품")
        add("1039", 71, "확정계약: 구속력 있는 약정")
        add("1039", 72, "예상거래: 예상되는 거래")
        add("1039", 73, "위험회피수단: 지정한 파생상품")
        add("1039", 74, "위험회피대상항목: 다음을 충족하는 항목")
        add("1039", 75, "⑴ 조건 하나")
        add("1039", 76, "⑵ 조건 둘")
        add("1039", 77, "위험회피효과: 상쇄되는 정도")
        return rows


if __name__ == "__main__":
    unittest.main()
