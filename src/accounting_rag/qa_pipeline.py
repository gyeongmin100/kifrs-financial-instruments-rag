from __future__ import annotations

from typing import Any, Mapping, Sequence

from accounting_rag.generation.answer import AnswerConfig, prepare_evidence_catalog
from accounting_rag.generation.citation_verifier import verify_citations


def _insufficient_answer(report: Mapping[str, Any]) -> dict[str, Any]:
    semantic = report.get("semantic")
    missing = semantic.get("missing_aspects", []) if isinstance(semantic, Mapping) else []
    if missing:
        detail = "; ".join(str(item) for item in missing if str(item).strip())
    else:
        failed = [
            name for name, passed in report.get("deterministic", {}).get("checks", {}).items()
            if not passed
        ]
        detail = ", ".join(failed) if failed else str(report.get("status", "unknown"))
    return {
        "conclusion": f"근거 부족: 현재 검색 결과만으로 질문에 신뢰성 있게 답하기 어렵습니다. ({detail})",
        "reasoning": ["근거 부족: 필요한 직접 근거와 질문 항목의 충족 여부를 확인하지 못했습니다."],
        "evidence": [],
    }


class AccountingQAPipeline:
    """Retrieve, gate, answer, and verify a grounded K-IFRS response."""

    def __init__(self, retriever: Any, sufficiency_checker: Any,
                 answer_generator: Any, *, max_retries: int = 1,
                 question_analyzer: Any | None = None) -> None:
        if max_retries not in {0, 1}:
            raise ValueError("max_retries must be 0 or 1")
        self.retriever = retriever
        self.sufficiency_checker = sufficiency_checker
        self.answer_generator = answer_generator
        self.max_retries = max_retries
        self.question_analyzer = question_analyzer

    @staticmethod
    def _evidence_from_catalog(
        catalog: Sequence[Mapping[str, Any]], evidence_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        by_id = {str(item["evidence_id"]): item for item in catalog}
        result: list[dict[str, Any]] = []
        for evidence_id in dict.fromkeys(str(value) for value in evidence_ids):
            item = by_id.get(evidence_id)
            if item is None:
                continue
            evidence: dict[str, Any] = {
                "evidence_id": evidence_id,
                "citation": str(item["citation"]),
                "statement": str(item["statement"]),
            }
            for key in (
                "source_id", "pdf_page_start", "pdf_page_end", "graph_path",
                "graph_hop", "candidate_source",
            ):
                if item.get(key) is not None:
                    evidence[key] = item[key]
            result.append(evidence)
        return result

    @classmethod
    def _enrich_answer_evidence(
        cls, answer: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        enriched = dict(answer)
        evidence = answer.get("evidence", [])
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            return enriched
        ids = [
            str(item.get("evidence_id") or "")
            for item in evidence if isinstance(item, Mapping)
        ]
        metadata_by_id = {
            item["evidence_id"]: item for item in cls._evidence_from_catalog(catalog, ids)
        }
        enriched["evidence"] = [
            {**dict(item), **{
                key: value for key, value in metadata_by_id.get(
                    str(item.get("evidence_id") or ""), {}
                ).items()
                if key not in {"evidence_id", "citation", "statement"}
            }}
            for item in evidence if isinstance(item, Mapping)
        ]
        return enriched

    @staticmethod
    def _retry_query(question: str, sufficiency: Mapping[str, Any]) -> str | None:
        semantic = sufficiency.get("semantic")
        if not isinstance(semantic, Mapping):
            return None
        missing = [str(item).strip() for item in semantic.get("missing_aspects", [])]
        missing = [item for item in missing if item]
        if not missing:
            return None
        return question + "\n추가로 확인할 사항: " + "; ".join(missing)

    def ask(self, question: str, *, standard_id: str | None = None,
            zone: str | None = None, top_k: int | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        analysis = self.question_analyzer.analyze(question) if self.question_analyzer else None
        analysis_dict = analysis.to_dict() if analysis is not None else None
        if standard_id is None and analysis is not None and len(analysis.requested_standard_ids) == 1:
            standard_id = analysis.requested_standard_ids[0]
        base_search_question = analysis.search_query if analysis is not None else question
        search_question = base_search_question
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.max_retries + 1):
            retrieval = self.retriever.retrieve(
                search_question, standard_id=standard_id, zone=zone, top_k=top_k
            )
            candidates = list(retrieval.get("results") or [])
            sufficiency = self.sufficiency_checker.check(question, candidates)
            attempts.append({
                "attempt": attempt + 1,
                "search_question": search_question,
                "result_count": len(candidates),
                "sufficiency_status": sufficiency.get("status"),
            })
            if sufficiency.get("sufficient"):
                break
            retry_query = self._retry_query(base_search_question, sufficiency)
            if attempt >= self.max_retries or retry_query is None:
                break
            search_question = retry_query
        retrieval = dict(retrieval)
        retrieval["attempts"] = attempts
        if not sufficiency.get("sufficient"):
            return {
                "status": "insufficient",
                "answer": _insufficient_answer(sufficiency),
                "retrieval": retrieval,
                "sufficiency": sufficiency,
                "citation_verification": None,
                "analysis": analysis_dict,
            }

        supported = set(sufficiency["semantic"]["supported_evidence_ids"])
        grounded_candidates = [
            item for item in candidates if str(item.get("chunk_id")) in supported
        ]
        if not grounded_candidates:
            return {
                "status": "insufficient",
                "answer": _insufficient_answer({"status": "no_supported_evidence"}),
                "retrieval": retrieval,
                "sufficiency": sufficiency,
                "citation_verification": None,
                "analysis": analysis_dict,
            }

        catalog = prepare_evidence_catalog(
            grounded_candidates,
            getattr(self.answer_generator, "config", AnswerConfig()),
        )
        answer = self.answer_generator.generate(question, grounded_candidates)
        answer = self._enrich_answer_evidence(answer, catalog)
        verification = verify_citations(answer, catalog).to_dict()
        generation_failed = answer.get("conclusion", "").startswith("근거 부족:")
        status = "answered" if verification["valid"] and not generation_failed else (
            "insufficient" if generation_failed else "generation_failed"
        )
        if status == "generation_failed":
            answer = {
                "conclusion": "근거 부족: 생성된 답변의 인용 무결성을 확인하지 못했습니다.",
                "reasoning": ["근거 부족: 검증되지 않은 답변은 사용자에게 제공하지 않습니다."],
                "evidence": [],
            }
        return {
            "status": status,
            "answer": answer,
            "retrieval": retrieval,
            "sufficiency": sufficiency,
            "citation_verification": verification,
            "analysis": analysis_dict,
        }
