import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.graph.loader import (
    batches,
    build_next_edges,
    build_page_edges,
    build_structure_rows,
    build_subparagraph_rows,
    section_id,
)


class GraphLoaderTests(unittest.TestCase):
    def paragraph(self, paragraph_id="P1", order=1, path=None):
        return {
            "paragraph_id": paragraph_id,
            "standard_id": "1109",
            "zone": "standard_body",
            "section_path": path or ["금융상품", "분류"],
            "document_order": order,
            "subitems": [{"marker": "⑴", "text": "조건", "xml_index": 2}],
            "pdf_page_start": 10,
            "pdf_page_end": 11,
            "page_match_confidence": 0.75,
        }

    def test_batches_preserve_all_rows(self) -> None:
        result = list(batches([{"n": n} for n in range(5)], 2))
        self.assertEqual([len(group) for group in result], [2, 2, 1])

    def test_section_ids_are_stable_and_zone_scoped(self) -> None:
        first = section_id("1109", "standard_body", ["A", "B"])
        self.assertEqual(first, section_id("1109", "standard_body", ["A", "B"]))
        self.assertEqual(first, section_id("1109", "application_guidance", ["A", "B"]))

    def test_structure_builds_hierarchy_and_paragraph_edge(self) -> None:
        rows = build_structure_rows([self.paragraph()])
        self.assertEqual(len(rows["sections"]), 2)
        self.assertEqual(len(rows["section_edges"]), 1)
        self.assertEqual(rows["paragraph_section_edges"][0]["paragraph_id"], "P1")

    def test_subparagraph_id_matches_chunk_source_id(self) -> None:
        nodes, edges = build_subparagraph_rows([self.paragraph()])
        self.assertEqual(nodes[0]["subparagraph_id"], "P1-S01")
        self.assertEqual(edges[0]["subparagraph_id"], "P1-S01")

    def test_page_edges_cover_inclusive_range(self) -> None:
        edges = build_page_edges([self.paragraph()], "paragraph_id")
        self.assertEqual([row["page_id"] for row in edges], ["KIFRS1109-PAGE-0010", "KIFRS1109-PAGE-0011"])

    def test_next_edges_follow_document_order(self) -> None:
        rows = [self.paragraph("P2", 2), self.paragraph("P1", 1)]
        self.assertEqual(build_next_edges(rows, "paragraph_id"), [{"source_id": "P1", "target_id": "P2"}])


if __name__ == "__main__":
    unittest.main()
