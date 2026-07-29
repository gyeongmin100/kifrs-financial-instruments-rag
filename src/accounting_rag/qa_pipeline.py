from __future__ import annotations

from typing import Any, Sequence


class AccountingQAPipeline:
    """Reject an empty retrieval; otherwise return the grounded model answer."""

    def __init__(self, retriever: Any, answer_generator: Any) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

    def ask(self, question: str, *, standard_id: str | None = None,
            zone: str | None = None, top_k: int | None = None,
            semantic_query: str | None = None,
            keyword_query: str | None = None,
            image_urls: Sequence[str] = ()) -> dict[str, Any]:
        question = question.strip()
        if not question and not image_urls:
            raise ValueError("question must not be empty")

        retrieval_options = {
            "standard_id": standard_id, "zone": zone, "top_k": top_k,
        }
        if keyword_query is not None:
            retrieval_options["keyword_query"] = keyword_query
        retrieval = self.retriever.retrieve(
            (semantic_query or question).strip(), **retrieval_options,
        )
        candidates = list(retrieval.get("results") or [])
        if not candidates:
            threshold_filtered = bool(retrieval.get("threshold_filtered_all"))
            return {
                "status": "insufficient",
                "reason": (
                    "no_supported_evidence" if threshold_filtered
                    else "no_evidence_found"
                ),
                "answer": {
                    "conclusion": (
                        "근거 부족: 질문과 충분히 관련된 기준서 문단을 찾지 못했습니다."
                        if threshold_filtered else
                        "근거 부족: 질문과 관련된 기준서 문단을 찾지 못했습니다."
                    ),
                    "reasoning": [
                        "근거 부족: 검색 후보가 관련성 기준을 충족하지 못했습니다."
                        if threshold_filtered else
                        "근거 부족: 검색 결과가 없어 답변할 수 없습니다."
                    ],
                    "evidence": [],
                },
                "retrieval": retrieval,
            }

        answer = self.answer_generator.generate(
            question, candidates, image_urls=image_urls,
        )
        # 임계값을 통과한 청크가 있으면 모델 응답을 그대로 전달한다. 일부 근거 부족이나
        # 실제 관련성 판단을 접두사·인용 개수로 다시 분류해 답변을 가리지 않는다.
        return {
            "status": "answered",
            "reason": "evidence_available",
            "answer": answer,
            "retrieval": retrieval,
        }
