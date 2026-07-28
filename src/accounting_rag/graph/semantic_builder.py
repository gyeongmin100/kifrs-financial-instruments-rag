from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from accounting_rag.graph.loader import read_jsonl


EXTRACTOR_VERSION = "official-definitions-v1"
_SPACE = re.compile(r"\s+")
_ALIAS = re.compile(r"이하\s*[‘']([^’']{2,60})[’'](?:이?라|로)\s*한다")
_LEADING_MARKER = re.compile(r"^[⑴-⑽㈎-㈛①-⑳]\s*")
_SOURCE_IDS = {
    "paragraph": "paragraph_id",
    "block": "block_id",
    "table": "table_id",
}


def normalize_text(value: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def concept_key(term: str) -> str:
    value = normalize_text(term).lower()
    value = re.sub(r"[\s·:：()'‘’\-]", "", value)
    return value


def concept_id(term: str) -> str:
    digest = hashlib.sha256(concept_key(term).encode("utf-8")).hexdigest()[:20]
    return f"CONCEPT-{digest}"


def _definition(term: str, definition: str, source_type: str, source_id: str,
                standard_id: str) -> dict:
    return {
        "term": normalize_text(term),
        "definition": normalize_text(definition),
        "source_type": source_type,
        "source_id": source_id,
        "standard_id": str(standard_id),
    }


def _table_definitions(tables: Iterable[dict]) -> list[dict]:
    result = []
    for table in tables:
        if table.get("zone") != "appendix_definitions":
            continue
        cells = {(int(cell["row"]), int(cell["column"])): cell.get("text", "")
                 for cell in table.get("cells", [])}
        for row in sorted({position[0] for position in cells}):
            term = normalize_text(cells.get((row, 0), ""))
            definition = normalize_text(cells.get((row, 1), ""))
            if term and definition:
                result.append(_definition(term, definition, "table", table["table_id"],
                                          table["standard_id"]))
    return result


def _colon_definition(blocks: list[dict], start: int, end: int) -> list[dict]:
    selected = [row for row in blocks if start <= int(row["document_order"]) <= end]
    selected.sort(key=lambda row: int(row["document_order"]))
    result = []
    current: dict | None = None
    for block in selected:
        text = normalize_text(block.get("text", ""))
        match = re.match(r"^([^:：]{2,60})[:：]\s*(.+)$", text)
        if match:
            if current:
                result.append(current)
            current = _definition(match.group(1), match.group(2), "block",
                                  block["block_id"], block["standard_id"])
        elif current and text:
            current["definition"] = normalize_text(f"{current['definition']} {text}")
    if current:
        result.append(current)
    return result


def extract_official_definitions(blocks: list[dict], tables: list[dict]) -> list[dict]:
    definitions = _table_definitions(tables)
    by_id = {row["block_id"]: row for row in blocks}

    # K-IFRS 1032 paragraph 11: three nested definitions followed by three colon definitions.
    definitions.append(_definition("금융상품", by_id["KIFRS1032-BLK-00083"]["text"].split(":", 1)[1],
                                   "block", "KIFRS1032-BLK-00083", "1032"))
    for term, start, end in (("금융자산", 84, 95), ("금융부채", 96, 106)):
        rows = [row for row in blocks if row.get("standard_id") == "1032"
                and start <= int(row["document_order"]) <= end]
        rows.sort(key=lambda row: int(row["document_order"]))
        definition = " ".join(normalize_text(row.get("text", "")) for row in rows)
        definitions.append(_definition(term, definition, "block", rows[0]["block_id"], "1032"))
    definitions.extend(_colon_definition(
        [row for row in blocks if row.get("standard_id") == "1032"], 107, 109
    ))

    # K-IFRS 1039 paragraph 9. Subitems after a colon remain part of that definition.
    definitions.extend(_colon_definition(
        [row for row in blocks if row.get("standard_id") == "1039"], 71, 77
    ))
    return definitions


def _listed_aliases(blocks: Iterable[dict], known_keys: set[str]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    ranges = {"1032": (111, 124), "1107": (560, 584), "1109": (660, 665)}
    for block in blocks:
        standard_id = str(block.get("standard_id"))
        if standard_id not in ranges:
            continue
        order = int(block["document_order"])
        if not ranges[standard_id][0] <= order <= ranges[standard_id][1]:
            continue
        term = _LEADING_MARKER.sub("", normalize_text(block.get("text", "")))
        key = concept_key(term)
        if key in known_keys:
            aliases[key].add(term)
        for alias in _ALIAS.findall(term):
            alias_key = concept_key(alias)
            if alias_key in known_keys:
                aliases[alias_key].add(alias)
                long_form = normalize_text(term.split("(이하", 1)[0])
                if long_form:
                    aliases[alias_key].add(long_form)
    return aliases


def _source_text(row: dict, source_type: str) -> str:
    if source_type != "table":
        return normalize_text(row.get("text", ""))
    cells = sorted(row.get("cells", []), key=lambda cell: (cell["row"], cell["column"]))
    return normalize_text(" ".join(cell.get("text", "") for cell in cells))


def build_semantic_kg(processed_dir: Path) -> dict[str, list[dict]]:
    paragraphs = read_jsonl(processed_dir / "paragraphs.jsonl")
    blocks = read_jsonl(processed_dir / "blocks.jsonl")
    tables = read_jsonl(processed_dir / "tables.jsonl")
    definitions = extract_official_definitions(blocks, tables)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in definitions:
        grouped[concept_key(row["term"])].append(row)
    listed_aliases = _listed_aliases(blocks, set(grouped))

    concepts = []
    definition_sources: set[tuple[str, str, str]] = set()
    for key, rows in sorted(grouped.items()):
        canonical = rows[0]["term"]
        aliases = {row["term"] for row in rows} | listed_aliases.get(key, set())
        for row in rows:
            definition_sources.add((row["source_type"], row["source_id"], concept_id(canonical)))
        definitions_json = json.dumps(
            [{"text": row["definition"], "standard_id": row["standard_id"],
              "source_type": row["source_type"], "source_id": row["source_id"]}
             for row in rows], ensure_ascii=False, sort_keys=True,
        )
        concepts.append({
            "concept_id": concept_id(canonical),
            "canonical_name": canonical,
            "aliases": sorted(alias for alias in aliases if alias != canonical),
            "alias_search_text": " ".join(sorted(aliases)),
            "definition": rows[0]["definition"],
            "definitions_json": definitions_json,
            "standard_ids": sorted({row["standard_id"] for row in rows}),
            "source_ids": sorted({row["source_id"] for row in rows}),
            "extractor_model": "deterministic",
            "extractor_version": EXTRACTOR_VERSION,
            "review_status": "approved",
            "managed_by_semantic_builder": True,
        })

    sources = [("paragraph", row) for row in paragraphs]
    sources += [("block", row) for row in blocks]
    sources += [("table", row) for row in tables]
    mentions = []
    for source_type, source in sources:
        text = _source_text(source, source_type)
        if not text:
            continue
        source_id = source[_SOURCE_IDS[source_type]]
        for concept in concepts:
            terms = [concept["canonical_name"], *concept["aliases"]]
            matches = [(text.find(term), term) for term in terms if term and text.find(term) >= 0]
            if not matches:
                continue
            start, span = min(matches, key=lambda item: (item[0], -len(item[1])))
            role = "definition" if (source_type, source_id, concept["concept_id"]) in definition_sources else "mention"
            raw_id = f"{source_type}\x1f{source_id}\x1f{concept['concept_id']}"
            mentions.append({
                "mention_id": "MENTION-" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24],
                "source_type": source_type,
                "source_id": source_id,
                "standard_id": str(source["standard_id"]),
                "concept_id": concept["concept_id"],
                "role": role,
                "source_text_span": span,
                "char_start": start,
                "char_end": start + len(span),
                "method": "official_definition" if role == "definition" else "lexical_exact",
                "provenance": "official_definition_parser" if role == "definition" else "lexical_concept_linker",
                "confidence": 1.0,
                "review_status": "approved",
                "extractor_version": EXTRACTOR_VERSION,
            })
    return {"concepts": concepts, "mentions": mentions}


def write_semantic_kg(output_dir: Path, graph: dict[str, list[dict]]) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("concepts", "mentions"):
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as target:
            for row in graph[name]:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {name: len(graph[name]) for name in ("concepts", "mentions")}
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
