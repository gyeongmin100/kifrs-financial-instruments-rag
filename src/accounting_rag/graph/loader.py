from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sequence


STANDARD_TITLES = {
    "1032": "금융상품: 표시",
    "1039": "금융상품: 인식과 측정",
    "1107": "금융상품: 공시",
    "1109": "금융상품",
}
SOURCE_LABELS = {"block": "Block", "table": "Table", "footnote": "Footnote"}
MANAGED_RELATIONSHIP_PROVENANCE = (
    "structure_parser",
    "document_order",
    "subitem_order",
    "pdf_page_order",
    "pdf_page_mapper",
    "chunk_builder",
    "explicit_reference_parser",
)
STRUCTURE_LABELS = {
    "paragraphs": ("Paragraph", "paragraph_id"),
    "blocks": ("Block", "block_id"),
    "tables": ("Table", "table_id"),
    "footnotes": ("Footnote", "footnote_id"),
    "chunks": ("Chunk", "chunk_id"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def batches(rows: Sequence[dict], batch_size: int) -> Iterator[list[dict]]:
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def _properties(row: dict, excluded: set[str] | None = None) -> dict:
    excluded = excluded or set()
    result = {}
    for key, value in row.items():
        if key in excluded or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            result[key] = value
    return result


def section_id(standard_id: str, zone: str, path: Sequence[str]) -> str:
    del zone  # Section IDs follow the schema's normalized path rule.
    normalized_path = "/".join(part.strip() for part in path)
    key = f"{standard_id}\x1f{normalized_path}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"KIFRS{standard_id}-SEC-{digest}"


def build_structure_rows(paragraphs: list[dict]) -> dict[str, list[dict]]:
    standards = [
        {
            "standard_id": standard_id,
            "properties": {
                "standard_id": standard_id,
                "name": f"K-IFRS 제{standard_id}호",
                "title": title,
            },
        }
        for standard_id, title in STANDARD_TITLES.items()
    ]
    zone_keys = sorted({(row["standard_id"], row["zone"]) for row in paragraphs})
    zones = [
        {
            "zone_id": f"KIFRS{standard_id}-Z-{zone}",
            "standard_id": standard_id,
            "properties": {
                "zone_id": f"KIFRS{standard_id}-Z-{zone}",
                "standard_id": standard_id,
                "type": zone,
            },
        }
        for standard_id, zone in zone_keys
    ]

    sections: dict[str, dict] = {}
    section_edges: set[tuple[str, str]] = set()
    zone_section_edges: set[tuple[str, str]] = set()
    paragraph_section_edges: list[dict] = []
    paragraph_zone_edges: list[dict] = []
    for paragraph in paragraphs:
        standard_id, zone = paragraph["standard_id"], paragraph["zone"]
        path = paragraph.get("section_path") or []
        previous_id = None
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            current_id = section_id(standard_id, zone, prefix)
            sections[current_id] = {
                "section_id": current_id,
                "properties": {
                    "section_id": current_id,
                    "standard_id": standard_id,
                    "zone": zone,
                    "title": prefix[-1],
                    "level": depth,
                    "path": prefix,
                    "path_key": "/".join(part.strip() for part in prefix),
                },
            }
            if previous_id:
                section_edges.add((previous_id, current_id))
            else:
                zone_section_edges.add(
                    (f"KIFRS{standard_id}-Z-{zone}", current_id)
                )
            previous_id = current_id
        if previous_id:
            paragraph_section_edges.append(
                {"section_id": previous_id, "paragraph_id": paragraph["paragraph_id"]}
            )
        else:
            paragraph_zone_edges.append(
                {
                    "zone_id": f"KIFRS{standard_id}-Z-{zone}",
                    "paragraph_id": paragraph["paragraph_id"],
                }
            )
    return {
        "standards": standards,
        "zones": zones,
        "sections": list(sections.values()),
        "standard_zone_edges": [
            {"standard_id": standard_id, "zone_id": f"KIFRS{standard_id}-Z-{zone}"}
            for standard_id, zone in zone_keys
        ],
        "zone_section_edges": [
            {"zone_id": source, "section_id": target}
            for source, target in sorted(zone_section_edges)
        ],
        "section_edges": [
            {"source_id": source, "target_id": target}
            for source, target in sorted(section_edges)
        ],
        "paragraph_section_edges": paragraph_section_edges,
        "paragraph_zone_edges": paragraph_zone_edges,
    }


def build_subparagraph_rows(paragraphs: list[dict]) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    for paragraph in paragraphs:
        for position, item in enumerate(paragraph.get("subitems", []), start=1):
            subparagraph_id = f"{paragraph['paragraph_id']}-S{position:02d}"
            nodes.append(
                {
                    "subparagraph_id": subparagraph_id,
                    "properties": {
                        "subparagraph_id": subparagraph_id,
                        "standard_id": paragraph["standard_id"],
                        "paragraph_id": paragraph["paragraph_id"],
                        "order": position,
                        "marker": item["marker"],
                        "text": item["text"],
                        "xml_index": item["xml_index"],
                    },
                }
            )
            edges.append(
                {
                    "paragraph_id": paragraph["paragraph_id"],
                    "subparagraph_id": subparagraph_id,
                }
            )
    return nodes, edges


def build_page_edges(rows: list[dict], id_key: str) -> list[dict]:
    edges = []
    for row in rows:
        start, end = row.get("pdf_page_start"), row.get("pdf_page_end")
        if start is None or end is None:
            continue
        for page in range(start, end + 1):
            edges.append(
                {
                    "source_id": row[id_key],
                    "page_id": f"KIFRS{row['standard_id']}-PAGE-{page:04d}",
                    "confidence": row.get("page_match_confidence", 0.0),
                }
            )
    return edges


def build_next_edges(rows: list[dict], id_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["standard_id"]].append(row)
    edges = []
    for standard_rows in grouped.values():
        ordered = sorted(
            standard_rows,
            key=lambda row: (
                row.get("document_order") if row.get("document_order") is not None else 10**9,
                row.get("part_index", 0),
                row[id_key],
            ),
        )
        edges.extend(
            {"source_id": source[id_key], "target_id": target[id_key]}
            for source, target in zip(ordered, ordered[1:])
        )
    return edges


def _node_rows(rows: list[dict], id_key: str, excluded: set[str] | None = None) -> list[dict]:
    return [
        {id_key: row[id_key], "properties": _properties(row, excluded)} for row in rows
    ]


def _run(session, query: str, rows: list[dict], batch_size: int) -> None:
    for group in batches(rows, batch_size):
        session.run(query, rows=group).consume()


def apply_schema(session, schema_path: Path) -> None:
    text = schema_path.read_text(encoding="utf-8")
    statements = []
    current = []
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            continue
        current.append(line)
        if ";" in line:
            statement = "\n".join(current).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            current = []
    trailing = "\n".join(current).strip()
    if trailing:
        statements.append(trailing)
    for statement in statements:
        session.run(statement).consume()


def load_graph(
    processed_dir: Path,
    chunks_path: Path,
    driver,
    database: str,
    schema_path: Path,
    batch_size: int = 500,
) -> dict:
    paragraphs = read_jsonl(processed_dir / "paragraphs.jsonl")
    blocks = read_jsonl(processed_dir / "blocks.jsonl")
    tables = read_jsonl(processed_dir / "tables.jsonl")
    footnotes = read_jsonl(processed_dir / "footnotes.jsonl")
    pdf_pages = read_jsonl(processed_dir / "pdf_pages.jsonl")
    references = read_jsonl(processed_dir / "references.jsonl")
    chunks = read_jsonl(chunks_path)
    document_manifest = json.loads(
        (processed_dir / "document_manifest.json").read_text(encoding="utf-8")
    )
    structure = build_structure_rows(paragraphs)
    manifest_by_standard = {
        row["standard_id"]: row for row in document_manifest["standards"]
    }
    for standard in structure["standards"]:
        source = manifest_by_standard[standard["standard_id"]]
        standard["properties"].update(
            {
                "source_file": source["source_file"],
                "source_sha256": source["source_sha256"],
                "paragraph_count": source["paragraphs"],
                "block_count": source["blocks"],
                "table_count": source["tables"],
                "footnote_count": source["footnotes"],
                "reference_count": source["references"],
            }
        )
    subparagraphs, paragraph_subparagraph_edges = build_subparagraph_rows(paragraphs)

    with driver.session(database=database) as session:
        apply_schema(session, schema_path)
        session.run(
            "MATCH ()-[r]->() WHERE r.provenance IN $provenance DELETE r",
            provenance=list(MANAGED_RELATIONSHIP_PROVENANCE),
        ).consume()
        _run(session, "UNWIND $rows AS row MERGE (n:Standard {standard_id: row.standard_id}) SET n += row.properties", structure["standards"], batch_size)
        _run(session, "UNWIND $rows AS row MERGE (n:Zone {zone_id: row.zone_id}) SET n += row.properties", structure["zones"], batch_size)
        _run(session, "UNWIND $rows AS row MERGE (n:Section {section_id: row.section_id}) SET n += row.properties", structure["sections"], batch_size)
        session.run(
            "MATCH (n:Section) WHERE NOT n.section_id IN $section_ids DETACH DELETE n",
            section_ids=[row["section_id"] for row in structure["sections"]],
        ).consume()

        node_specs = [
            ("Paragraph", "paragraph_id", paragraphs, {"subitems", "block_ids"}),
            ("Subparagraph", "subparagraph_id", subparagraphs, None),
            ("Block", "block_id", blocks, {"references", "table_ids", "footnote_ids"}),
            ("Table", "table_id", tables, {"cells"}),
            ("Footnote", "footnote_id", footnotes, None),
            ("Chunk", "chunk_id", chunks, None),
        ]
        for label, id_key, source_rows, excluded in node_specs:
            rows = source_rows if label == "Subparagraph" else _node_rows(source_rows, id_key, excluded)
            _run(
                session,
                f"UNWIND $rows AS row MERGE (n:{label} {{{id_key}: row.{id_key}}}) SET n += row.properties",
                rows,
                batch_size,
            )

        table_json_rows = [
            {
                "table_id": row["table_id"],
                "cells_json": json.dumps(row["cells"], ensure_ascii=False),
                "serialized_text": " ".join(
                    cell["text"] for cell in row["cells"] if cell.get("text")
                ),
            }
            for row in tables
        ]
        _run(session, "UNWIND $rows AS row MATCH (n:Table {table_id: row.table_id}) SET n.cells_json = row.cells_json, n.serialized_text = row.serialized_text", table_json_rows, batch_size)
        page_rows = [
            {
                "page_id": f"KIFRS{row['standard_id']}-PAGE-{row['pdf_page']:04d}",
                "properties": {
                    **row,
                    "page_id": f"KIFRS{row['standard_id']}-PAGE-{row['pdf_page']:04d}",
                },
            }
            for row in pdf_pages
        ]
        _run(session, "UNWIND $rows AS row MERGE (n:PdfPage {page_id: row.page_id}) SET n += row.properties", page_rows, batch_size)

        approved = (
            "SET r.provenance = 'structure_parser', r.confidence = 1.0, "
            "r.review_status = 'approved'"
        )
        relation_queries = [
            (structure["standard_zone_edges"], f"MATCH (a:Standard {{standard_id: row.standard_id}}), (b:Zone {{zone_id: row.zone_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}"),
            (structure["zone_section_edges"], f"MATCH (a:Zone {{zone_id: row.zone_id}}), (b:Section {{section_id: row.section_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}"),
            (structure["section_edges"], f"MATCH (a:Section {{section_id: row.source_id}}), (b:Section {{section_id: row.target_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}"),
            (structure["paragraph_section_edges"], f"MATCH (a:Section {{section_id: row.section_id}}), (b:Paragraph {{paragraph_id: row.paragraph_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}"),
            (structure["paragraph_zone_edges"], f"MATCH (a:Zone {{zone_id: row.zone_id}}), (b:Paragraph {{paragraph_id: row.paragraph_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}"),
            (paragraph_subparagraph_edges, f"MATCH (a:Paragraph {{paragraph_id: row.paragraph_id}}), (b:Subparagraph {{subparagraph_id: row.subparagraph_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}"),
        ]
        for rows, match_query in relation_queries:
            _run(session, f"UNWIND $rows AS row {match_query}", rows, batch_size)

        parent_specs = [
            (blocks, "Block", "block_id", "HAS_BLOCK"),
            (tables, "Table", "table_id", "HAS_TABLE"),
            (footnotes, "Footnote", "footnote_id", "HAS_FOOTNOTE"),
        ]
        for source_rows, label, id_key, relation in parent_specs:
            rows = [
                {"paragraph_id": row["parent_paragraph_id"], "target_id": row[id_key]}
                for row in source_rows
                if row.get("parent_paragraph_id")
            ]
            _run(session, f"UNWIND $rows AS row MATCH (a:Paragraph {{paragraph_id: row.paragraph_id}}), (b:{label} {{{id_key}: row.target_id}}) MERGE (a)-[r:{relation}]->(b) {approved}", rows, batch_size)

        standard_page_edges = [
            {
                "standard_id": row["properties"]["standard_id"],
                "page_id": row["page_id"],
            }
            for row in page_rows
        ]
        _run(session, f"UNWIND $rows AS row MATCH (a:Standard {{standard_id: row.standard_id}}), (b:PdfPage {{page_id: row.page_id}}) MERGE (a)-[r:CONTAINS]->(b) {approved}", standard_page_edges, batch_size)

        for source_rows, label, id_key, source_list_key, relation in (
            (blocks, "Table", "table_id", "table_ids", "HAS_TABLE"),
            (blocks, "Footnote", "footnote_id", "footnote_ids", "HAS_FOOTNOTE"),
        ):
            rows = [
                {"block_id": block["block_id"], "target_id": target_id}
                for block in source_rows
                for target_id in block[source_list_key]
            ]
            _run(session, f"UNWIND $rows AS row MATCH (a:Block {{block_id: row.block_id}}), (b:{label} {{{id_key}: row.target_id}}) MERGE (a)-[r:{relation}]->(b) {approved}", rows, batch_size)

        page_sources = [
            ("Paragraph", "paragraph_id", paragraphs),
            ("Block", "block_id", blocks),
            ("Table", "table_id", tables),
            ("Footnote", "footnote_id", footnotes),
            ("Chunk", "chunk_id", chunks),
        ]
        for label, id_key, source_rows in page_sources:
            rows = build_page_edges(source_rows, id_key)
            _run(session, f"UNWIND $rows AS row MATCH (a:{label} {{{id_key}: row.source_id}}), (b:PdfPage {{page_id: row.page_id}}) MERGE (a)-[r:APPEARS_ON]->(b) SET r.confidence = row.confidence, r.provenance = 'pdf_page_mapper', r.review_status = 'rule_based'", rows, batch_size)

        for label, id_key, source_rows in (
            ("Paragraph", "paragraph_id", paragraphs),
            ("Chunk", "chunk_id", chunks),
        ):
            rows = build_next_edges(source_rows, id_key)
            _run(session, f"UNWIND $rows AS row MATCH (a:{label} {{{id_key}: row.source_id}}), (b:{label} {{{id_key}: row.target_id}}) MERGE (a)-[r:NEXT]->(b) SET r.provenance = 'document_order', r.confidence = 1.0, r.review_status = 'approved'", rows, batch_size)

        subparagraph_next = []
        grouped_subparagraphs: dict[str, list[dict]] = defaultdict(list)
        for row in subparagraphs:
            grouped_subparagraphs[row["properties"]["paragraph_id"]].append(row)
        for siblings in grouped_subparagraphs.values():
            ordered = sorted(siblings, key=lambda row: row["properties"]["order"])
            subparagraph_next.extend(
                {"source_id": source["subparagraph_id"], "target_id": target["subparagraph_id"]}
                for source, target in zip(ordered, ordered[1:])
            )
        _run(session, "UNWIND $rows AS row MATCH (a:Subparagraph {subparagraph_id: row.source_id}), (b:Subparagraph {subparagraph_id: row.target_id}) MERGE (a)-[r:NEXT]->(b) SET r.provenance = 'subitem_order', r.confidence = 1.0, r.review_status = 'approved'", subparagraph_next, batch_size)
        page_next = []
        grouped_pages: dict[str, list[dict]] = defaultdict(list)
        for row in page_rows:
            grouped_pages[row["properties"]["standard_id"]].append(row)
        for standard_pages in grouped_pages.values():
            ordered = sorted(standard_pages, key=lambda row: row["properties"]["pdf_page"])
            page_next.extend(
                {"source_id": source["page_id"], "target_id": target["page_id"]}
                for source, target in zip(ordered, ordered[1:])
            )
        _run(session, "UNWIND $rows AS row MATCH (a:PdfPage {page_id: row.source_id}), (b:PdfPage {page_id: row.target_id}) MERGE (a)-[r:NEXT]->(b) SET r.provenance = 'pdf_page_order', r.confidence = 1.0, r.review_status = 'approved'", page_next, batch_size)

        derived_specs = [
            ("Paragraph", "paragraph_id", "source_paragraph_ids"),
            ("Subparagraph", "subparagraph_id", "source_subparagraph_ids"),
            ("Block", "block_id", "block_ids"),
            ("Table", "table_id", "table_ids"),
            ("Footnote", "footnote_id", "footnote_ids"),
        ]
        for label, id_key, source_key in derived_specs:
            rows = [
                {"chunk_id": chunk["chunk_id"], "target_id": target_id}
                for chunk in chunks
                for target_id in chunk[source_key]
            ]
            _run(session, f"UNWIND $rows AS row MATCH (a:Chunk {{chunk_id: row.chunk_id}}), (b:{label} {{{id_key}: row.target_id}}) MERGE (a)-[r:DERIVED_FROM]->(b) SET r.provenance = 'chunk_builder', r.confidence = 1.0, r.review_status = 'approved'", rows, batch_size)

        external_ids = sorted(
            {
                row["target_standard"]
                for row in references
                if row.get("target_standard")
                and row["target_standard"] not in STANDARD_TITLES
            }
        )
        external_rows = [
            {
                "external_standard_id": standard_id,
                "properties": {
                    "external_standard_id": standard_id,
                    "name": f"K-IFRS 제{standard_id}호",
                    "loaded": False,
                },
            }
            for standard_id in external_ids
        ]
        _run(session, "UNWIND $rows AS row MERGE (n:ExternalStandard {external_standard_id: row.external_standard_id}) SET n += row.properties", external_rows, batch_size)

        for source_type, source_label in SOURCE_LABELS.items():
            source_key = f"{source_type}_id"
            rows = []
            for reference in references:
                if (
                    reference["source_type"] != source_type
                    or reference["resolution_status"] not in {"resolved", "resolved_range"}
                ):
                    continue
                for target_id in reference["resolved_target_ids"]:
                    rows.append(
                        {
                            "source_id": reference["source_id"],
                            "target_id": target_id,
                            "reference_id": reference["reference_id"],
                            "properties": _properties(
                                reference, {"resolved_target_ids"}
                            ),
                        }
                    )
            _run(session, f"UNWIND $rows AS row MATCH (a:{source_label} {{{source_key}: row.source_id}}), (b:Paragraph {{paragraph_id: row.target_id}}) MERGE (a)-[r:REFERS_TO {{reference_id: row.reference_id}}]->(b) SET r += row.properties, r.provenance = 'explicit_reference_parser', r.review_status = 'approved'", rows, batch_size)

            standard_rows = []
            external_standard_rows = []
            for reference in references:
                if (
                    reference["source_type"] != source_type
                    or reference["resolution_status"] not in {"resolved_standard", "external_standard"}
                    or not reference.get("target_standard")
                ):
                    continue
                target = {
                    "source_id": reference["source_id"],
                    "target_id": reference["target_standard"],
                    "reference_id": reference["reference_id"],
                    "properties": _properties(reference, {"resolved_target_ids"}),
                }
                if reference["target_standard"] in STANDARD_TITLES:
                    standard_rows.append(target)
                else:
                    external_standard_rows.append(target)
            _run(session, f"UNWIND $rows AS row MATCH (a:{source_label} {{{source_key}: row.source_id}}), (b:Standard {{standard_id: row.target_id}}) MERGE (a)-[r:REFERS_TO {{reference_id: row.reference_id}}]->(b) SET r += row.properties, r.provenance = 'explicit_reference_parser', r.review_status = 'approved'", standard_rows, batch_size)
            _run(session, f"UNWIND $rows AS row MATCH (a:{source_label} {{{source_key}: row.source_id}}), (b:ExternalStandard {{external_standard_id: row.target_id}}) MERGE (a)-[r:REFERS_TO {{reference_id: row.reference_id}}]->(b) SET r += row.properties, r.provenance = 'explicit_reference_parser', r.review_status = 'approved'", external_standard_rows, batch_size)

        counts = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY label"
        ).data()
    return {
        "nodes": {row["label"]: row["count"] for row in counts},
        "source_counts": {
            "paragraphs": len(paragraphs),
            "subparagraphs": len(subparagraphs),
            "blocks": len(blocks),
            "tables": len(tables),
            "footnotes": len(footnotes),
            "pdf_pages": len(pdf_pages),
            "chunks": len(chunks),
            "references": len(references),
        },
    }
