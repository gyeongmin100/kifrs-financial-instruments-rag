from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.graph.loader import read_jsonl  # noqa: E402


def main() -> None:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(PROJECT_ROOT / ".env")
    concepts = read_jsonl(PROJECT_ROOT / "data" / "semantic" / "concepts.jsonl")
    mentions = read_jsonl(PROJECT_ROOT / "data" / "semantic" / "mentions.jsonl")
    expected_definitions = sum(row["role"] == "definition" for row in mentions)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.environ["NEO4J_DATABASE"]) as session:
            counts = session.run(
                "MATCH (c:Concept) OPTIONAL MATCH ()-[m:MENTIONS]->(c) "
                "RETURN count(DISTINCT c) AS concepts, count(m) AS mentions, "
                "count(CASE WHEN m.role = 'definition' THEN 1 END) AS definitions"
            ).single().data()
            invalid_edges = session.run(
                "MATCH ()-[m:MENTIONS]->(:Concept) "
                "WHERE m.review_status <> 'approved' OR m.source_text_span IS NULL "
                "OR m.provenance IS NULL RETURN count(m) AS count"
            ).single()["count"]
            missing_definitions = session.run(
                "MATCH (c:Concept) WHERE NOT ()-[:MENTIONS {role: 'definition'}]->(c) "
                "RETURN count(c) AS count"
            ).single()["count"]
            ecl_paths = session.run(
                "MATCH (p:Paragraph)-[:MENTIONS]->(c:Concept {canonical_name: '기대신용손실'})"
                "<-[d:MENTIONS {role: 'definition'}]-(source) "
                "RETURN p.paragraph_id AS paragraph_id, c.concept_id AS concept_id, "
                "coalesce(source.table_id, source.block_id) AS definition_source_id LIMIT 5"
            ).data()
            index = session.run(
                "SHOW INDEXES YIELD name, state WHERE name = 'concept_fulltext' RETURN state"
            ).single()
            probe = session.run(
                "CALL db.index.fulltext.queryNodes('concept_fulltext', '기대신용손실', {limit: 5}) "
                "YIELD node RETURN node.canonical_name AS name"
            ).data()
    finally:
        driver.close()

    expected = {"concepts": len(concepts), "mentions": len(mentions),
                "definitions": expected_definitions}
    valid = (
        counts == expected
        and invalid_edges == 0
        and missing_definitions == 0
        and bool(ecl_paths)
        and index is not None and index["state"] == "ONLINE"
        and any(row["name"] == "기대신용손실" for row in probe)
    )
    report = {
        "valid": valid,
        "expected": expected,
        "actual": counts,
        "invalid_edges": invalid_edges,
        "concepts_without_definition": missing_definitions,
        "concept_fulltext_state": index["state"] if index else None,
        "ecl_probe": probe,
        "ecl_definition_paths": ecl_paths,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
