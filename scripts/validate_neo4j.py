from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.graph.loader import (  # noqa: E402
    STANDARD_TITLES,
    build_structure_rows,
    build_subparagraph_rows,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the loaded K-IFRS Neo4j graph.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--chunks", type=Path, default=PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(PROJECT_ROOT / ".env")
    paragraphs = read_jsonl(args.processed_dir / "paragraphs.jsonl")
    blocks = read_jsonl(args.processed_dir / "blocks.jsonl")
    tables = read_jsonl(args.processed_dir / "tables.jsonl")
    footnotes = read_jsonl(args.processed_dir / "footnotes.jsonl")
    pages = read_jsonl(args.processed_dir / "pdf_pages.jsonl")
    references = read_jsonl(args.processed_dir / "references.jsonl")
    chunks = read_jsonl(args.chunks)
    structure = build_structure_rows(paragraphs)
    subparagraphs, _ = build_subparagraph_rows(paragraphs)
    external_standards = {
        row["target_standard"]
        for row in references
        if row.get("target_standard") and row["target_standard"] not in STANDARD_TITLES
    }
    expected = {
        "Standard": len(STANDARD_TITLES),
        "Zone": len(structure["zones"]),
        "Section": len(structure["sections"]),
        "Paragraph": len(paragraphs),
        "Subparagraph": len(subparagraphs),
        "Block": len(blocks),
        "Table": len(tables),
        "Footnote": len(footnotes),
        "PdfPage": len(pages),
        "Chunk": len(chunks),
        "ExternalStandard": len(external_standards),
    }

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            actual = {
                row["label"]: row["count"]
                for row in session.run(
                    "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count"
                )
            }
            missing_chunk_sources = session.run(
                "MATCH (c:Chunk) WHERE NOT (c)-[:DERIVED_FROM]->() RETURN count(c) AS count"
            ).single()["count"]
            missing_search_pages = session.run(
                "MATCH (c:Chunk {searchable: true}) WHERE NOT (c)-[:APPEARS_ON]->(:PdfPage) RETURN count(c) AS count"
            ).single()["count"]
            uncontained_paragraphs = session.run(
                "MATCH (p:Paragraph) WHERE NOT ()-[:CONTAINS]->(p) RETURN count(p) AS count"
            ).single()["count"]
            unapproved_references = session.run(
                "MATCH ()-[r:REFERS_TO]->() WHERE r.review_status <> 'approved' RETURN count(r) AS count"
            ).single()["count"]
            relationship_counts = {
                row["type"]: row["count"]
                for row in session.run(
                    "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
                )
            }
            index_rows = {
                row["name"]: {"state": row["state"], "options": row["options"]}
                for row in session.run(
                    "SHOW INDEXES YIELD name, state, options RETURN name, state, options"
                )
            }
            fulltext_probe = [
                row["text"]
                for row in session.run(
                    "CALL db.index.fulltext.queryNodes('chunk_fulltext', $probe, {limit: 20}) "
                    "YIELD node, score WHERE node.searchable = true AND node.inactive = false "
                    "RETURN node.text AS text ORDER BY score DESC",
                    probe="기대신용손실",
                )
            ]
    finally:
        driver.close()

    mismatches = {
        label: {"expected": count, "actual": actual.get(label, 0)}
        for label, count in expected.items()
        if actual.get(label, 0) != count
    }
    required_indexes = {"chunk_fulltext", "chunk_embedding_vector"}
    offline_indexes = {
        name: index_rows.get(name)
        for name in required_indexes
        if index_rows.get(name, {}).get("state") != "ONLINE"
    }
    fulltext_analyzer = (
        index_rows.get("chunk_fulltext", {})
        .get("options", {})
        .get("indexConfig", {})
        .get("fulltext.analyzer")
    )
    fulltext_probe_valid = any("기대신용손실" in text for text in fulltext_probe)
    report = {
        "valid": not mismatches
        and missing_chunk_sources == 0
        and missing_search_pages == 0
        and uncontained_paragraphs == 0
        and unapproved_references == 0
        and not offline_indexes,
        "fulltext_probe_valid": fulltext_probe_valid,
        "expected_nodes": expected,
        "actual_nodes": actual,
        "node_mismatches": mismatches,
        "relationship_counts": relationship_counts,
        "missing_chunk_sources": missing_chunk_sources,
        "missing_search_pages": missing_search_pages,
        "uncontained_paragraphs": uncontained_paragraphs,
        "unapproved_references": unapproved_references,
        "required_index_states": {
            name: index_rows.get(name, {}).get("state") for name in sorted(required_indexes)
        },
        "fulltext_analyzer": fulltext_analyzer,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    report["valid"] = report["valid"] and fulltext_probe_valid and fulltext_analyzer == "cjk"
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
