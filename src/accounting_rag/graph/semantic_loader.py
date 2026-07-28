from __future__ import annotations

from pathlib import Path

from accounting_rag.graph.loader import _run, apply_schema, read_jsonl


SOURCE_LABELS = {
    "paragraph": ("Paragraph", "paragraph_id"),
    "block": ("Block", "block_id"),
    "table": ("Table", "table_id"),
}


def load_semantic_graph(semantic_dir: Path, driver, database: str,
                        schema_path: Path, batch_size: int = 500) -> dict:
    concepts = read_jsonl(semantic_dir / "concepts.jsonl")
    mentions = read_jsonl(semantic_dir / "mentions.jsonl")
    concept_ids = [row["concept_id"] for row in concepts]

    with driver.session(database=database) as session:
        apply_schema(session, schema_path)
        _run(
            session,
            "UNWIND $rows AS row MERGE (n:Concept {concept_id: row.concept_id}) SET n += row",
            concepts,
            batch_size,
        )
        session.run(
            "MATCH (n:Concept {managed_by_semantic_builder: true}) "
            "WHERE NOT n.concept_id IN $concept_ids DETACH DELETE n",
            concept_ids=concept_ids,
        ).consume()
        session.run(
            "MATCH ()-[r:MENTIONS]->() WHERE r.provenance IN "
            "['official_definition_parser', 'lexical_concept_linker'] DELETE r"
        ).consume()
        for source_type, (label, id_key) in SOURCE_LABELS.items():
            rows = [row for row in mentions if row["source_type"] == source_type]
            _run(
                session,
                f"UNWIND $rows AS row MATCH (source:{label} {{{id_key}: row.source_id}}) "
                "MATCH (concept:Concept {concept_id: row.concept_id}) "
                "MERGE (source)-[r:MENTIONS {mention_id: row.mention_id}]->(concept) "
                "SET r += row",
                rows,
                batch_size,
            )
        counts = session.run(
            "MATCH (c:Concept) OPTIONAL MATCH ()-[m:MENTIONS]->(c) "
            "RETURN count(DISTINCT c) AS concepts, count(m) AS mentions, "
            "count(DISTINCT CASE WHEN m.role = 'definition' THEN m END) AS definitions"
        ).single()
    return dict(counts)
