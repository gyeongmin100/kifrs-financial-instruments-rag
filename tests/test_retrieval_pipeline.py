import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline  # noqa: E402


class FakeHybrid:
    def __init__(self, seeds):
        self.seeds = seeds
        self.calls = []

    def search(self, question, *, standard_id=None, zone=None, sparse_query=None):
        self.calls.append({
            "question": question, "standard_id": standard_id, "zone": zone,
            "sparse_query": sparse_query,
        })
        return [dict(seed) for seed in self.seeds]


class FakeScoredHybrid:
    def __init__(self, dense, sparse, results):
        self.snapshot = {"dense": dense, "sparse": sparse, "results": results}

    def search_with_scores(self, question, *, standard_id=None, zone=None):
        return self.snapshot


class FakeSession:
    def __init__(self, rows, recorder):
        self.rows = rows
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, _query, **parameters):
        self.recorder.append(parameters)
        return [dict(row) for row in self.rows]


class FakeDriver:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.queries = []

    def session(self, database=None):
        self.queries.append({"database": database})
        return FakeSession(self.rows, self.queries)


class RetrievalPipelineTests(unittest.TestCase):
    def test_returns_hybrid_seeds_tagged_with_their_source(self):
        hybrid = FakeHybrid([
            {"chunk_id": "C1", "text": "본문 1"},
            {"chunk_id": "C2", "text": "본문 2"},
        ])
        output = RetrievalPipeline(hybrid, FakeDriver(), database="neo4j").retrieve(
            " 기대신용손실은? ", standard_id="1109",
        )
        self.assertEqual([item["chunk_id"] for item in output["results"]], ["C1", "C2"])
        self.assertEqual({item["candidate_source"] for item in output["results"]}, {"hybrid"})
        self.assertEqual(output["seed_count"], 2)
        self.assertEqual(hybrid.calls[0]["question"], "기대신용손실은?")
        self.assertEqual(hybrid.calls[0]["standard_id"], "1109")

    def test_forwards_keyword_query_to_sparse_search(self):
        hybrid = FakeHybrid([{"chunk_id": "C1", "text": "본문"}])
        RetrievalPipeline(hybrid, FakeDriver(), database="neo4j").retrieve(
            "금융부채 조건변경의 회계처리",
            keyword_query="금융부채 조건변경 유효이자율",
        )

        self.assertEqual(
            hybrid.calls[0]["question"], "금융부채 조건변경의 회계처리",
        )
        self.assertEqual(
            hybrid.calls[0]["sparse_query"], "금융부채 조건변경 유효이자율",
        )

    def test_appends_sibling_chunks_of_a_paragraph_split_across_chunks(self):
        hybrid = FakeHybrid([
            {"chunk_id": "KIFRS1032-11-C01", "text": "금융자산은 다음의 자산을 말한다."},
        ])
        driver = FakeDriver([
            {"chunk_id": "KIFRS1032-11-C02", "text": "금융부채는 다음의 부채를 말한다."},
        ])
        output = RetrievalPipeline(hybrid, driver, database="neo4j").retrieve("금융자산의 정의")
        self.assertEqual(
            [item["chunk_id"] for item in output["results"]],
            ["KIFRS1032-11-C01", "KIFRS1032-11-C02"],
        )
        self.assertEqual(output["results"][1]["candidate_source"], "sibling")
        self.assertEqual(output["sibling_count"], 1)

    def test_sibling_lookup_skips_chunks_already_retrieved(self):
        hybrid = FakeHybrid([{"chunk_id": "C1", "text": "본문"}])
        driver = FakeDriver([{"chunk_id": "C1", "text": "본문"}])
        output = RetrievalPipeline(hybrid, driver, database="neo4j").retrieve("질문")
        self.assertEqual([item["chunk_id"] for item in output["results"]], ["C1"])
        self.assertEqual(output["sibling_count"], 0)

    def test_sibling_lookup_is_skipped_when_disabled(self):
        hybrid = FakeHybrid([{"chunk_id": "C1", "text": "본문"}])
        driver = FakeDriver([{"chunk_id": "C2", "text": "형제"}])
        output = RetrievalPipeline(
            hybrid, driver, database="neo4j", config=RetrievalConfig(max_siblings=0),
        ).retrieve("질문")
        self.assertEqual(output["sibling_count"], 0)
        self.assertEqual(driver.queries, [])

    def test_top_k_limits_the_seeds_passed_on(self):
        hybrid = FakeHybrid([{"chunk_id": f"C{index}", "text": "본문"} for index in range(20)])
        output = RetrievalPipeline(
            hybrid, FakeDriver(), database="neo4j", config=RetrievalConfig(max_siblings=0),
        ).retrieve("질문", top_k=3)
        self.assertEqual(output["result_count"], 3)

    def test_reports_when_thresholds_remove_all_raw_candidates(self):
        hybrid = FakeScoredHybrid(
            dense=[{"chunk_id": "D1", "score": 0.4}],
            sparse=[{"chunk_id": "S1", "score": 2.0}],
            results=[],
        )
        output = RetrievalPipeline(hybrid, FakeDriver()).retrieve("질문")
        self.assertEqual(output["raw_candidate_count"], 2)
        self.assertTrue(output["threshold_filtered_all"])
        self.assertEqual(output["results"], [])

    def test_empty_question_stops_before_dependencies(self):
        hybrid = FakeHybrid([])
        driver = FakeDriver()
        with self.assertRaises(ValueError):
            RetrievalPipeline(hybrid, driver).retrieve("  ")
        self.assertEqual(hybrid.calls, [])
        self.assertEqual(driver.queries, [])


class RetrievalConfigTests(unittest.TestCase):
    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(top_k=0)
        with self.assertRaises(ValueError):
            RetrievalConfig(max_siblings=-1)

    def test_reads_the_retrieval_section(self):
        config = RetrievalConfig.from_yaml(
            Path(__file__).resolve().parents[1] / "config" / "retrieval.yaml"
        )
        self.assertGreater(config.top_k, 0)
        self.assertGreaterEqual(config.max_siblings, 0)


if __name__ == "__main__":
    unittest.main()
