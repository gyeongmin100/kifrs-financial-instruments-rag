import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.qa_pipeline import AccountingQAPipeline


class StubRetriever:
    def __init__(self, results): self.results = results; self.calls = []
    def retrieve(self, question, **kwargs):
        self.calls.append((question, kwargs)); return {"results": self.results}


class StubChecker:
    def __init__(self, report):
        self.reports = list(report) if isinstance(report, list) else [report]
        self.calls = []
    def check(self, question, candidates):
        self.calls.append((question, candidates)); return self.reports.pop(0)


class StubGenerator:
    def __init__(self, answer): self.answer = answer; self.calls = []
    def generate(self, question, candidates):
        self.calls.append((question, candidates)); return self.answer


class StubAnalyzer:
    def analyze(self, question):
        return SimpleNamespace(
            requested_standard_ids=("1109",), search_query="분석된 검색어",
            to_dict=lambda: {"concepts": ["금융자산"], "search_query": "분석된 검색어"},
        )


def candidate(chunk_id="C1", citation="문단 1", **metadata):
    return {
        "chunk_id": chunk_id, "text": "근거", "citation_label": citation,
        **metadata,
    }


class QAPipelineTests(unittest.TestCase):
    @staticmethod
    def sufficient_report(chunk_id="C1"):
        return {
            "sufficient": True,
            "status": "sufficient",
            "semantic": {"supported_evidence_ids": [chunk_id], "missing_aspects": []},
        }

    def test_insufficient_skips_generation(self):
        generator = StubGenerator({})
        pipeline = AccountingQAPipeline(
            StubRetriever([candidate()]),
            StubChecker({"sufficient": False, "status": "deterministic_failed",
                         "deterministic": {"checks": {"citations": False}},
                         "semantic": None}),
            generator,
        )
        result = pipeline.ask("질문")
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(generator.calls, [])
        self.assertEqual(result["answer"]["evidence"], [])

    def test_supported_candidates_only_are_answered_and_verified(self):
        candidates = [
            candidate("C1", "문단 1"),
            candidate(
                "C2", "문단 2", pdf_page_start=41, pdf_page_end=42,
                graph_path=["C1", "REFERS_TO", "C2"], graph_hop=1,
                candidate_source="graph",
            ),
        ]
        checker = StubChecker({
            "sufficient": True,
            "semantic": {"supported_evidence_ids": ["C2"]},
        })
        answer = {
            "conclusion": "결론 [E1]",
            "reasoning": ["판단 [E1]"],
            "evidence": [{"evidence_id": "E1", "citation": "문단 2", "statement": "근거"}],
        }
        generator = StubGenerator(answer)
        result = AccountingQAPipeline(
            StubRetriever(candidates), checker, generator
        ).ask("질문", standard_id="1109")
        self.assertEqual(result["status"], "answered")
        self.assertEqual([row["chunk_id"] for row in generator.calls[0][1]], ["C2"])
        self.assertTrue(result["citation_verification"]["valid"])
        self.assertEqual(result["answer"]["evidence"][0]["source_id"], "C2")
        self.assertEqual(result["answer"]["evidence"][0]["pdf_page_start"], 41)
        self.assertEqual(result["answer"]["evidence"][0]["pdf_page_end"], 42)
        self.assertEqual(result["answer"]["evidence"][0]["graph_hop"], 1)
        self.assertEqual(
            result["answer"]["evidence"][0]["graph_path"],
            ["C1", "REFERS_TO", "C2"],
        )
        self.assertEqual(result["answer"]["evidence"][0]["candidate_source"], "graph")

    def test_invalid_generated_citation_is_not_exposed(self):
        report = {"sufficient": True, "semantic": {"supported_evidence_ids": ["C1"]}}
        bad = {"conclusion": "결론 [E1]", "reasoning": ["판단 [E1]"],
               "evidence": [{"evidence_id": "E1", "citation": "위조", "statement": "근거"}]}
        result = AccountingQAPipeline(
            StubRetriever([candidate()]), StubChecker(report), StubGenerator(bad)
        ).ask("질문")
        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(result["answer"]["evidence"], [])

    def test_missing_aspects_trigger_exactly_one_retry(self):
        retriever = StubRetriever([candidate("C1")])
        insufficient = {
            "sufficient": False, "status": "semantic_insufficient",
            "semantic": {"missing_aspects": ["예외 조건"]},
            "deterministic": {"checks": {}},
        }
        sufficient = {
            "sufficient": True,
            "status": "sufficient",
            "semantic": {"supported_evidence_ids": ["C1"], "missing_aspects": []},
        }
        answer = {"conclusion": "결론 [E1]", "reasoning": ["판단 [E1]"],
                  "evidence": [{"evidence_id": "E1", "citation": "문단 1",
                                "statement": "근거"}]}
        result = AccountingQAPipeline(
            retriever, StubChecker([insufficient, sufficient]), StubGenerator(answer)
        ).ask("원 질문")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(retriever.calls), 2)
        self.assertIn("예외 조건", retriever.calls[1][0])
        self.assertEqual(len(result["retrieval"]["attempts"]), 2)

    def test_question_analysis_drives_search_and_standard_filter(self):
        retriever = StubRetriever([candidate("C1")])
        report = {"sufficient": True, "semantic": {"supported_evidence_ids": ["C1"]}}
        answer = {"conclusion": "결론 [E1]", "reasoning": ["판단 [E1]"],
                  "evidence": [{"evidence_id": "E1", "citation": "문단 1",
                                "statement": "근거"}]}
        result = AccountingQAPipeline(
            retriever, StubChecker(report), StubGenerator(answer),
            question_analyzer=StubAnalyzer(),
        ).ask("원문 질문")
        self.assertEqual(retriever.calls[0][0], "분석된 검색어")
        self.assertEqual(retriever.calls[0][1]["standard_id"], "1109")
        self.assertEqual(result["analysis"]["concepts"], ["금융자산"])

    def test_multiple_choice_wording_uses_the_same_answer_generator(self):
        report = self.sufficient_report()
        answer = {"conclusion": "정답은 ①입니다. [E1]", "reasoning": ["판단 [E1]"],
                  "evidence": [{"evidence_id": "E1", "citation": "문단 1",
                                "statement": "근거"}]}
        generator = StubGenerator(answer)
        result = AccountingQAPipeline(
            StubRetriever([candidate()]), StubChecker(report), generator,
        ).ask("다음 중 옳은 것은? ① 보기 A ② 보기 B")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(generator.calls), 1)
        self.assertNotIn("mode_result", result)


if __name__ == "__main__":
    unittest.main()
