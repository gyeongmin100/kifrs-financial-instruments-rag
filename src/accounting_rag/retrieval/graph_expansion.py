from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


_SUPPORTED_EDGES = frozenset(
    {
        "REFERS_TO",
        "CONTAINS",
        "HAS_BLOCK",
        "HAS_TABLE",
        "HAS_FOOTNOTE",
        "NEXT",
        "MENTIONS",
    }
)
_TARGET_LABELS = (
    "Paragraph",
    "Subparagraph",
    "Block",
    "Table",
    "Footnote",
    "Section",
    "Standard",
    "ExternalStandard",
    "Concept",
)
_DETAIL_LABELS = frozenset({"Subparagraph", "Block", "Table", "Footnote"})
_BRIDGE_ONLY_LABELS = frozenset({"Concept"})
_TEXT_TOKEN = re.compile(r"[0-9A-Za-z가-힣]{2,}")


@dataclass(frozen=True)
class GraphExpansionConfig:
    default_hops: int = 1
    max_hops: int = 2
    max_nodes: int = 40
    max_frontier: int = 12
    edge_allowlist: tuple[str, ...] = tuple(sorted(_SUPPORTED_EDGES))
    edge_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "REFERS_TO": 1.0,
            "CONTAINS": 0.8,
            "HAS_BLOCK": 0.75,
            "HAS_TABLE": 0.8,
            "HAS_FOOTNOTE": 0.7,
            "NEXT": 0.45,
            "MENTIONS": 0.95,
        }
    )
    lexical_overlap_min: int = 1

    def __post_init__(self) -> None:
        if self.default_hops != 1:
            raise ValueError("default_hops must be 1")
        if self.max_hops not in {1, 2} or self.max_hops < self.default_hops:
            raise ValueError("max_hops must be 1 or 2")
        if self.max_nodes <= 0 or self.max_frontier <= 0:
            raise ValueError("max_nodes and max_frontier must be greater than zero")
        if self.lexical_overlap_min <= 0:
            raise ValueError("lexical_overlap_min must be greater than zero")
        edges = set(self.edge_allowlist)
        unknown = edges - _SUPPORTED_EDGES
        if unknown:
            raise ValueError(f"Unsupported graph expansion edges: {sorted(unknown)}")
        if not edges:
            raise ValueError("edge_allowlist must not be empty")
        if set(self.edge_weights) - _SUPPORTED_EDGES:
            raise ValueError("edge_weights contains an unsupported edge")
        if any(weight < 0 for weight in self.edge_weights.values()):
            raise ValueError("edge weights must be non-negative")

    @classmethod
    def from_yaml(cls, path: Path) -> "GraphExpansionConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = raw.get("graph", raw)
        defaults = cls()
        return cls(
            default_hops=int(values.get("default_hops", defaults.default_hops)),
            max_hops=int(values.get("max_hops", defaults.max_hops)),
            max_nodes=int(values.get("max_nodes", defaults.max_nodes)),
            max_frontier=int(values.get("max_frontier", defaults.max_frontier)),
            edge_allowlist=tuple(values.get("edge_allowlist", defaults.edge_allowlist)),
            edge_weights={**defaults.edge_weights, **values.get("edge_weights", {})},
            lexical_overlap_min=int(
                values.get("second_hop", {}).get(
                    "lexical_overlap_min", defaults.lexical_overlap_min
                )
            ),
        )


def question_terms(question: str) -> tuple[str, ...]:
    """Return stable, low-noise terms for the optional second-hop gate."""
    return tuple(dict.fromkeys(token.lower() for token in _TEXT_TOKEN.findall(question)))


def _node_id(row: Mapping[str, Any], prefix: str) -> str:
    value = row.get(f"{prefix}_id")
    if value is None:
        raise ValueError(f"Neo4j graph row is missing {prefix}_id")
    return str(value)


def _labels(row: Mapping[str, Any], prefix: str) -> tuple[str, ...]:
    return tuple(str(label) for label in row.get(f"{prefix}_labels", ()))


def _overlap_count(terms: Sequence[str], row: Mapping[str, Any]) -> int:
    haystack = " ".join(
        str(row.get(key) or "") for key in ("target_text", "target_title", "target_number")
    ).lower()
    return sum(term in haystack for term in terms)


class GraphExpander:
    """Expand Hybrid RAG seed chunks through a bounded, provenance-preserving graph."""

    _SOURCE_QUERY = """
UNWIND $seeds AS seed
MATCH (chunk:Chunk {chunk_id: seed.chunk_id})-[edge:DERIVED_FROM]->(source)
WHERE edge.review_status = 'approved'
RETURN seed.chunk_id AS seed_chunk_id, seed.seed_rank AS seed_rank,
       seed.seed_score AS seed_score, elementId(source) AS source_element_id,
       coalesce(source.paragraph_id, source.subparagraph_id, source.block_id,
                source.table_id, source.footnote_id) AS source_id,
       labels(source) AS source_labels, source.text AS source_text,
       source.number AS source_number, source.standard_id AS source_standard_id,
       source.zone AS source_zone,
       properties(edge) AS edge_properties
ORDER BY seed_rank, source_id
""".strip()

    _NEIGHBOR_QUERY = """
UNWIND $frontier AS item
MATCH (source) WHERE elementId(source) = item.element_id
MATCH (source)-[edge]-(target)
WHERE type(edge) IN $edge_allowlist
  AND edge.review_status = 'approved'
  AND (type(edge) <> 'REFERS_TO' OR startNode(edge) = source)
  AND (NOT 'Concept' IN labels(source) OR type(edge) <> 'MENTIONS' OR edge.role = 'definition')
  AND any(label IN labels(target) WHERE label IN $target_labels)
RETURN item.frontier_key AS frontier_key, elementId(source) AS source_element_id,
       coalesce(source.paragraph_id, source.subparagraph_id, source.block_id,
                source.table_id, source.footnote_id, source.section_id, source.concept_id,
                source.standard_id, source.external_standard_id) AS source_id,
       labels(source) AS source_labels, elementId(target) AS target_element_id,
       coalesce(target.paragraph_id, target.subparagraph_id, target.block_id,
                target.table_id, target.footnote_id, target.section_id, target.concept_id,
                target.standard_id, target.external_standard_id) AS target_id,
       labels(target) AS target_labels,
       coalesce(target.text, target.definition, target.serialized_text) AS target_text,
       coalesce(target.title, target.canonical_name, target.table_id, target.block_id) AS target_title,
       target.number AS target_number,
       target.standard_id AS target_standard_id, target.zone AS target_zone,
       type(edge) AS edge_type,
       CASE WHEN startNode(edge) = source THEN 'outgoing' ELSE 'incoming' END AS direction,
       properties(edge) AS edge_properties
ORDER BY frontier_key, edge_type, target_id
""".strip()

    def __init__(self, driver: Any, *, database: str,
                 config: GraphExpansionConfig | None = None) -> None:
        self.driver = driver
        self.database = database
        self.config = config or GraphExpansionConfig()

    @staticmethod
    def _rows(result: Any) -> list[dict[str, Any]]:
        return [record.data() if hasattr(record, "data") else dict(record) for record in result]

    def _neighbors(self, session: Any, frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not frontier:
            return []
        return self._rows(
            session.run(
                self._NEIGHBOR_QUERY,
                frontier=frontier,
                edge_allowlist=list(self.config.edge_allowlist),
                target_labels=list(_TARGET_LABELS),
            )
        )

    def expand(self, question: str,
               seeds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if not seeds:
            return []

        seed_rows = [
            {
                "chunk_id": str(seed["chunk_id"]),
                "seed_rank": rank,
                "seed_score": float(seed.get("rrf_score", 0.0)),
            }
            for rank, seed in enumerate(seeds, start=1)
        ]
        terms = question_terms(question)
        evidence: dict[str, dict[str, Any]] = {}
        source_paths: dict[str, dict[str, Any]] = {}

        with self.driver.session(database=self.database) as session:
            source_rows = self._rows(session.run(self._SOURCE_QUERY, seeds=seed_rows))
            for index, row in enumerate(source_rows):
                source_id = _node_id(row, "source")
                path = [{
                    "edge_type": "DERIVED_FROM",
                    "direction": "outgoing",
                    "from_id": str(row["seed_chunk_id"]),
                    "to_id": source_id,
                    **dict(row.get("edge_properties") or {}),
                }]
                key = f"source:{index}"
                source_paths[key] = {
                    "seed_chunk_id": str(row["seed_chunk_id"]),
                    "seed_rank": int(row["seed_rank"]),
                    "seed_score": float(row["seed_score"]),
                    "element_id": str(row["source_element_id"]),
                    "node_id": source_id,
                    "node_labels": _labels(row, "source"),
                    "path": path,
                }
                self._merge_evidence(
                    evidence, source_id, _labels(row, "source"), row.get("source_text"),
                    row.get("source_number"), row.get("source_standard_id"),
                    row.get("source_zone"), 0, float(row["seed_score"]), path,
                    str(row["seed_chunk_id"]),
                )

            first_frontier = [
                {"frontier_key": key, "element_id": value["element_id"]}
                for key, value in source_paths.items()
            ]
            first_rows = self._neighbors(session, first_frontier)
            second_candidates: list[tuple[float, dict[str, Any]]] = []
            first_paths: dict[str, dict[str, Any]] = {}
            for index, row in enumerate(first_rows):
                parent = source_paths[str(row["frontier_key"])]
                target_id = _node_id(row, "target")
                edge_type = str(row["edge_type"])
                edge = {
                    "edge_type": edge_type,
                    "direction": str(row["direction"]),
                    "from_id": _node_id(row, "source"),
                    "to_id": target_id,
                    **dict(row.get("edge_properties") or {}),
                }
                path = [*parent["path"], edge]
                overlap = _overlap_count(terms, row)
                score = self._path_score(parent["seed_score"], [edge_type], overlap)
                target_labels = _labels(row, "target")
                if not set(target_labels) & _BRIDGE_ONLY_LABELS:
                    self._merge_evidence(
                        evidence, target_id, target_labels, row.get("target_text"),
                        row.get("target_number") or row.get("target_title"),
                        row.get("target_standard_id"), row.get("target_zone"),
                        1, score, path,
                        parent["seed_chunk_id"],
                    )
                path_key = f"first:{index}"
                first_paths[path_key] = {
                    **parent,
                    "element_id": str(row["target_element_id"]),
                    "node_id": target_id,
                    "node_labels": target_labels,
                    "path": path,
                    "edge_types": [edge_type],
                }
                if self.config.max_hops == 2 and self._needs_second_hop(
                    parent["node_labels"], target_labels, edge_type, overlap
                ):
                    second_candidates.append((score, {"frontier_key": path_key,
                                                       "element_id": str(row["target_element_id"])}))

            second_candidates.sort(key=lambda item: (-item[0], item[1]["frontier_key"]))
            second_frontier = [item[1] for item in second_candidates[:self.config.max_frontier]]
            for row in self._neighbors(session, second_frontier):
                parent = first_paths[str(row["frontier_key"])]
                target_id = _node_id(row, "target")
                if target_id in {step["from_id"] for step in parent["path"]}:
                    continue
                edge_type = str(row["edge_type"])
                edge = {
                    "edge_type": edge_type,
                    "direction": str(row["direction"]),
                    "from_id": _node_id(row, "source"),
                    "to_id": target_id,
                    **dict(row.get("edge_properties") or {}),
                }
                path = [*parent["path"], edge]
                overlap = _overlap_count(terms, row)
                score = self._path_score(
                    parent["seed_score"], [*parent["edge_types"], edge_type], overlap
                )
                target_labels = _labels(row, "target")
                if not set(target_labels) & _BRIDGE_ONLY_LABELS:
                    self._merge_evidence(
                        evidence, target_id, target_labels, row.get("target_text"),
                        row.get("target_number") or row.get("target_title"),
                        row.get("target_standard_id"), row.get("target_zone"),
                        2, score, path,
                        parent["seed_chunk_id"],
                    )

        return sorted(
            evidence.values(), key=lambda item: (-item["graph_score"], item["node_id"])
        )[: self.config.max_nodes]

    def _path_score(self, seed_score: float, edge_types: Sequence[str], overlap: int) -> float:
        relation_score = 1.0
        for edge_type in edge_types:
            relation_score *= float(self.config.edge_weights.get(edge_type, 0.0))
        return seed_score * relation_score * (1.0 + min(overlap, 3) * 0.1)

    def _needs_second_hop(self, source_labels: Sequence[str], target_labels: Sequence[str],
                          edge_type: str, overlap: int) -> bool:
        if edge_type in {"REFERS_TO", "MENTIONS"}:
            return True
        if overlap >= self.config.lexical_overlap_min:
            return True
        return bool(set(source_labels) & _DETAIL_LABELS and "Paragraph" in target_labels)

    @staticmethod
    def _merge_evidence(evidence: dict[str, dict[str, Any]], node_id: str,
                        labels: Sequence[str], text: Any, citation: Any,
                        standard_id: Any, zone: Any, hop: int, score: float,
                        path: list[dict[str, Any]], seed_chunk_id: str) -> None:
        item = evidence.setdefault(
            node_id,
            {
                "node_id": node_id,
                "node_labels": list(labels),
                "text": text,
                "citation": citation,
                "standard_id": standard_id,
                "zone": zone,
                "hop": hop,
                "graph_score": score,
                "seed_chunk_ids": [],
                "paths": [],
            },
        )
        item["hop"] = min(item["hop"], hop)
        item["graph_score"] = max(item["graph_score"], score)
        if seed_chunk_id not in item["seed_chunk_ids"]:
            item["seed_chunk_ids"].append(seed_chunk_id)
        if path not in item["paths"]:
            item["paths"].append(path)
