from __future__ import annotations

from typing import Any

# 답변 생성기가 근거로 답할 수 없다고 판단하면 결론을 이 접두사로 시작한다.
# 별도의 충분성 판정 단계를 두는 대신 이 신호로 근거 부족을 구분한다.
_INSUFFICIENT_PREFIXES = ("근거 부족:", "답변 불가:")


class AccountingQAPipeline:
    """Retrieve grounded chunks and let the answer generator decide if they suffice."""

    def __init__(self, retriever: Any, answer_generator: Any) -> None:
        self.retriever = retriever
        self.answer_generator = answer_generator

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

        answer = self.answer_generator.generate(question, candidates)
        conclusion = str(answer.get("conclusion") or "")
        # 인용한 근거가 하나도 없으면 접두사 표현과 무관하게 근거 부족으로 본다.
        # 접두사만 믿으면 모델이 다른 말로 거절했을 때 답변으로 잘못 분류된다.
        declined = (
            not (answer.get("evidence") or [])
            or conclusion.startswith(_INSUFFICIENT_PREFIXES)
        )
        return {
            "status": "insufficient" if declined else "answered",
            "reason": "self_declined" if declined else "sufficient",
            "answer": answer,
            "retrieval": retrieval,
        }
