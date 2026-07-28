import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.retrieval.embeddings import (
    EmbeddingConfig,
    build_embedding_cache,
    embed_texts,
    load_embeddings_neo4j,
    read_jsonl,
    searchable_inputs,
    validate_cache,
)


class FakeEmbeddings:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        data = [
            SimpleNamespace(index=index, embedding=[float(index)] * self.dimensions)
            for index, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(data)))


class FakeClient:
    def __init__(self, dimensions):
        self.embeddings = FakeEmbeddings(dimensions)


class FakeResult:
    def __init__(self, updated):
        self.updated = updated

    def single(self):
        return {"updated": self.updated}


class FakeSession:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return FakeResult(len(parameters["rows"]))


class FakeDriver:
    def __init__(self):
        self.opened_database = None
        self.fake_session = FakeSession()

    def session(self, database):
        self.opened_database = database
        return self.fake_session


def chunk(chunk_id="C1", text="context", searchable=True):
    return {
        "chunk_id": chunk_id,
        "contextualized_text": text,
        "searchable": searchable,
    }


class EmbeddingTests(unittest.TestCase):
    def test_searchable_inputs_only_uses_contextualized_text(self):
        rows = searchable_inputs([chunk(), chunk("C2", "ignored", False)])
        self.assertEqual([row["chunk_id"] for row in rows], ["C1"])
        self.assertEqual(rows[0]["text"], "context")

    def test_empty_searchable_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty contextualized_text"):
            searchable_inputs([chunk(text="  ")])

    def test_embed_texts_sends_array_and_restores_index_order(self):
        config = EmbeddingConfig("model", dimensions=3, batch_size=2)
        client = FakeClient(3)
        vectors = embed_texts(client, ["a", "b"], config)
        self.assertEqual(vectors, [[0.0] * 3, [1.0] * 3])
        self.assertEqual(
            client.embeddings.calls[0],
            {
                "input": ["a", "b"],
                "model": "model",
                "dimensions": 3,
                "encoding_format": "float",
            },
        )

    def test_build_cache_resumes_and_reembeds_changed_text(self):
        config = EmbeddingConfig("model", dimensions=3, batch_size=1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.jsonl"
            first_client = FakeClient(3)
            first = build_embedding_cache(
                [chunk("C1", "one"), chunk("C2", "two")], path, first_client, config
            )
            self.assertEqual(first["created"], 2)
            second_client = FakeClient(3)
            second = build_embedding_cache(
                [chunk("C1", "one"), chunk("C2", "changed")], path, second_client, config
            )
            self.assertEqual(second["cached_before"], 1)
            self.assertEqual(second["created"], 1)
            self.assertEqual(len(second_client.embeddings.calls), 1)
            rows = read_jsonl(path)
            self.assertEqual([row["chunk_id"] for row in rows], ["C1", "C2"])
            self.assertTrue(validate_cache(
                [chunk("C1", "one"), chunk("C2", "changed")], rows, config
            )["valid"])

    def test_bad_vector_dimension_is_rejected(self):
        config = EmbeddingConfig("model", dimensions=3)
        client = FakeClient(2)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            embed_texts(client, ["a"], config)

    def test_neo4j_loader_uses_unwind_and_sets_metadata(self):
        driver = FakeDriver()
        rows = [
            {"chunk_id": "C1", "model": "m", "dimensions": 3, "text_sha256": "h1", "embedding": [0.0] * 3},
            {"chunk_id": "C2", "model": "m", "dimensions": 3, "text_sha256": "h2", "embedding": [1.0] * 3},
        ]
        report = load_embeddings_neo4j(rows, driver, "neo4j", batch_size=1)
        self.assertTrue(report["valid"])
        self.assertEqual(len(driver.fake_session.calls), 2)
        query = driver.fake_session.calls[0][0]
        self.assertIn("UNWIND $rows", query)
        self.assertIn("embedding_text_sha256", query)


if __name__ == "__main__":
    unittest.main()
