from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from accounting_rag.generation.answer import OpenAIAnswerGenerator  # noqa: E402
from accounting_rag.generation.sufficiency import (  # noqa: E402
    EvidenceSufficiencyChecker, OpenAISemanticJudge, SufficiencyConfig,
)
from accounting_rag.qa_pipeline import AccountingQAPipeline  # noqa: E402
from accounting_rag.query.analysis import OpenAIQuestionAnalyzer, QuestionAnalysisConfig  # noqa: E402
from accounting_rag.retrieval.graph_expansion import GraphExpander, GraphExpansionConfig  # noqa: E402
from accounting_rag.retrieval.hybrid import HybridConfig, HybridRetriever  # noqa: E402
from accounting_rag.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from accounting_rag.retrieval.reranker import OpenAIReranker, RerankConfig  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask a grounded K-IFRS question.")
    parser.add_argument("question")
    parser.add_argument("--standard-id")
    parser.add_argument("--zone")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env")
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
                "OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL", "OPENAI_RERANK_MODEL",
                "OPENAI_CHAT_MODEL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    retrieval_config = PROJECT_ROOT / "config" / "retrieval.yaml"
    answer_config = PROJECT_ROOT / "config" / "answering.yaml"
    try:
        retriever = RetrievalPipeline(
            HybridRetriever(driver, client, database=os.environ["NEO4J_DATABASE"],
                            embedding_model=os.environ["OPENAI_EMBEDDING_MODEL"],
                            config=HybridConfig.from_yaml(retrieval_config)),
            GraphExpander(driver, database=os.environ["NEO4J_DATABASE"],
                          config=GraphExpansionConfig.from_yaml(retrieval_config)),
            OpenAIReranker(client, model=os.environ["OPENAI_RERANK_MODEL"],
                           config=RerankConfig.from_yaml(retrieval_config)),
        )
        sufficiency_config = SufficiencyConfig.from_yaml(answer_config)
        checker = EvidenceSufficiencyChecker(
            OpenAISemanticJudge(
                client, model=os.environ["OPENAI_CHAT_MODEL"],
                max_evidence_chars=sufficiency_config.max_evidence_chars,
                max_output_tokens=sufficiency_config.max_output_tokens,
            ),
            config=sufficiency_config,
        )
        pipeline = AccountingQAPipeline(
            retriever, checker,
            OpenAIAnswerGenerator(client, model=os.environ["OPENAI_CHAT_MODEL"]),
            question_analyzer=OpenAIQuestionAnalyzer(
                client,
                model=os.environ["OPENAI_CHAT_MODEL"],
                config=QuestionAnalysisConfig.from_yaml(PROJECT_ROOT / "config" / "query.yaml"),
            ),
        )
        result = pipeline.ask(args.question, standard_id=args.standard_id,
                              zone=args.zone, top_k=args.top_k)
    finally:
        driver.close()
    if args.debug:
        retrieval = result["retrieval"]
        output = {
            "status": result["status"],
            "answer": result["answer"],
            "retrieval": {key: value for key, value in retrieval.items() if key != "results"},
            "sufficiency": result["sufficiency"],
            "citation_verification": result["citation_verification"],
            "analysis": result["analysis"],
        }
    else:
        output = result["answer"]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
