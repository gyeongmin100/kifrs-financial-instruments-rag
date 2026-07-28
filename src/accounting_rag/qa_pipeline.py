from __future__ import annotations

from typing import Any, Mapping, Sequence

from accounting_rag.generation.answer import AnswerConfig, prepare_evidence_catalog

# 답변 생성기가 근거로 답할 수 없다고 판단하면 결론을 이 접두사로 시작한다.
# 별도의 충분성 판정 단계를 두는 대신 이 신호 하나로 근거 부족을 구분한다.
_INSUFFICIENT_PREFIXES = ("근거 부족:", "답변 불가:")


class AccountingQAPipeline:
    """Retrieve grounded chunks and let the answer generator decide if they suffice."""

    def __init__(self, retriever: Any, answer_generator: Any) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

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
            for key in ("source_id", "pdf_page_start", "pdf_page_end", "candidate_source"):
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

    def ask(self, question: str, *, standard_id: str | None = None,
            zone: str | None = None, top_k: int | None = None) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        retrieval = self.retriever.retrieve(
            question, standard_id=standard_id, zone=zone, top_k=top_k
        )
        candidates = list(retrieval.get("results") or [])
        if not candidates:
            return {
                "status": "insufficient",
                "reason": "no_evidence_found",
                "answer": {
                    "conclusion": "근거 부족: 질문과 관련된 기준서 문단을 찾지 못했습니다.",
                    "reasoning": ["근거 부족: 검색 결과가 없어 답변할 수 없습니다."],
                    "evidence": [],
                },
                "retrieval": retrieval,
            }

        catalog = prepare_evidence_catalog(
            candidates, getattr(self.answer_generator, "config", AnswerConfig()),
        )
        answer = self._enrich_answer_evidence(
            self.answer_generator.generate(question, candidates), catalog
        )
        conclusion = str(answer.get("conclusion") or "")
        declined = conclusion.startswith(_INSUFFICIENT_PREFIXES)
        return {
            "status": "insufficient" if declined else "answered",
            "reason": "self_declined" if declined else "sufficient",
            "answer": answer,
            "retrieval": retrieval,
        }
