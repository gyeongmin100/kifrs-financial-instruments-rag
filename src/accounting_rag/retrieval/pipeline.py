from __future__ import annotations

from typing import Any, Mapping, Sequence

from accounting_rag.retrieval.hybrid import normalize_standard_id


def _graph_candidate(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """Convert one graph node into the candidate shape expected by the reranker."""
    hop = int(evidence.get("hop", 0))
    text = str(evidence.get("text") or "").strip()
    node_id = str(evidence.get("node_id") or "").strip()
    if hop < 1 or not text or not node_id:
        return None

    paths = list(evidence.get("paths") or [])
    labels = [str(label) for label in evidence.get("node_labels") or []]
    return {
        "chunk_id": f"GRAPH::{node_id}",
        "candidate_source": "graph",
        "text": text,
        "standard_id": evidence.get("standard_id"),
        "zone": evidence.get("zone"),
        "chunk_type": labels[0] if labels else None,
        "citation_label": evidence.get("citation"),
        "source_channels": ["graph"],
        "graph_node_id": node_id,
        "graph_path": paths[0] if paths else [],
        "graph_paths": paths,
        "graph_hop": hop,
        "graph_distance": hop,
        "graph_score": float(evidence.get("graph_score", 0.0)),
        "graph_seed_chunk_ids": list(evidence.get("seed_chunk_ids") or []),
    }


def build_rerank_candidates(
    seeds: Sequence[Mapping[str, Any]],
    graph_evidence: Sequence[Mapping[str, Any]],
    *,
    standard_id: str | None = None,
    zone: str | None = None,
) -> list[dict[str, Any]]:
    """Combine hybrid seeds and textual graph evidence without candidate ID collisions."""
    normalized_standard = normalize_standard_id(standard_id)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for seed in seeds:
        candidate = dict(seed)
        chunk_id = str(candidate.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen:
            continue
        candidate["chunk_id"] = chunk_id
        candidate["candidate_source"] = "hybrid"
        candidates.append(candidate)
        seen.add(chunk_id)

    for evidence in graph_evidence:
        candidate = _graph_candidate(evidence)
        if candidate is None:
            continue
        if normalized_standard is not None and str(candidate.get("standard_id")) != normalized_standard:
            continue
        if zone is not None and candidate.get("zone") != zone:
            continue
        if candidate["chunk_id"] in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate["chunk_id"])
    return candidates


class RetrievalPipeline:
    """Run Hybrid seed retrieval, bounded graph expansion, and OpenAI reranking."""

    def __init__(self, hybrid_retriever: Any, graph_expander: Any, reranker: Any) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.graph_expander = graph_expander
        self.reranker = reranker

    def retrieve(
        self,
        question: str,
        *,
        standard_id: str | None = None,
        zone: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        normalized_standard = normalize_standard_id(standard_id)

        seeds = self.hybrid_retriever.search(
            question, standard_id=normalized_standard, zone=zone
        )
        graph_evidence = self.graph_expander.expand(question, seeds)
        candidates = build_rerank_candidates(
            seeds,
            graph_evidence,
            standard_id=normalized_standard,
            zone=zone,
        )
        results = self.reranker.rerank(question, candidates, top_k=top_k)
        return {
            "question": question,
            "filters": {"standard_id": normalized_standard, "zone": zone},
            "seed_count": len(seeds),
            "graph_evidence_count": len(graph_evidence),
            "candidate_count": len(candidates),
            "result_count": len(results),
            "results": results,
        }
