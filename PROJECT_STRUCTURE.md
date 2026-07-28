# Accounting RAG 프로젝트 구조

## 1. 프로젝트 정의

### 프로젝트명

**K-IFRS Financial Instruments QA**

K-IFRS 제1032호·제1039호·제1107호·제1109호를 근거로 금융상품 관련 일반 질문, 기준서 해석 질문, 비교 질문, 계산 질문 및 시험형 질문에 답하는 최종제품 수준의 질의응답 시스템이다.

OpenAI API와 Neo4j 하나를 사용하며, 검색은 키워드·벡터 기반 Hybrid RAG로 관련 문단을 먼저 찾은 뒤 해당 문단을 시작점으로 GraphRAG를 수행하는 직렬 구조를 사용한다. 최종 답변은 반드시 기준서 원문 문단으로 추적할 수 있어야 한다.

### 제품 목표

1. 사용자의 자연어 표현이 기준서 용어와 달라도 관련 문단을 찾는다.
2. 한 문단만 제시하지 않고 참조 문단, 정의, 조건, 예외, 적용지침과 적용사례를 연결한다.
3. 답변을 `결론 / 판단 과정 / 근거` 형식으로 제공한다.
4. 존재하지 않는 문단을 인용하거나 검색 근거 밖의 내용을 단정하지 않는다.
5. 파싱부터 답변까지 모든 중간 결과가 원문 HWPX와 PDF 페이지로 역추적된다.

### 대상 기준서

```text
금융상품
├─ 제1109호: 금융상품의 인식·분류·측정·손상·제거
├─ 제1032호: 금융상품의 부채·자본 구분과 표시
├─ 제1107호: 금융상품 공시
└─ 제1039호: 현재 적용 가능한 위험회피회계
```

네 기준서에 명시적으로 언급된 외부 기준서는 참조 사실과 식별자만 보존한다. 원문이 적재되지 않은 외부 기준서의 내용을 알고 있는 것처럼 답하지 않는다.

### 평가 범위

자동 벤치마크, 네 가지 RAG 방식 비교, CPA 문제 수집·정답률 계산은 현재 구현 범위에서 제외한다. 사용자가 완성된 시스템에 질문을 입력하여 직접 평가한다. 시스템은 수동 검토를 돕기 위해 검색 Seed, 그래프 경로, 재정렬 결과, 인용 문단과 오류 로그를 남긴다.

---

## 2. 확정된 설계 결정

| 항목 | 확정 내용 | 근거 |
|---|---|---|
| 데이터베이스 | Neo4j 하나 사용 | 구조 그래프, 벡터 인덱스, 전문 검색 인덱스를 한 저장소에서 관리할 수 있음 |
| 개발 환경 | 로컬 Neo4j로 개발 후 필요 시 Aura로 이전 | 로컬에서 반복 적재·삭제·검증하기 쉽고, 스키마와 Cypher를 유지한 채 배포 환경을 바꿀 수 있음 |
| 생성 모델 | `.env`의 `OPENAI_CHAT_MODEL` | 코드 하드코딩 없이 모델 교체 가능 |
| 임베딩 모델 | `.env`의 `OPENAI_EMBEDDING_MODEL` | 임베딩 차원과 인덱스를 설정으로 고정 |
| 재정렬 모델 | `OPENAI_RERANK_MODEL=gpt-5.6-terra` | 별도 공급자 없이 OpenAI Structured Output으로 근거 묶음을 평가 |
| 검색 구조 | Hybrid → Graph 확장 직렬 구조 | 자연어로 먼저 관련 문단을 찾은 뒤 정확한 구조·의미 관계를 확장하는 편이 엔터티 직접 연결 실패에 강함 |
| Hybrid 결합 | Dense + Sparse를 RRF로 결합 | 서로 다른 점수 체계를 직접 비교하지 않고 순위를 결합 |
| Graph 시작점 | Hybrid 상위 결과를 Seed로 사용 | 관련 없는 노드에서 그래프가 확장되는 위험을 줄임 |
| 예외 경로 | 정확한 기준서·문단번호는 해당 노드를 직접 조회 | 명확한 식별자를 다시 의미 검색할 필요가 없음 |
| 재정렬 단위 | 개별 Chunk가 아닌 그래프 근거 묶음 | Seed와 조건·예외·참조 관계를 함께 평가하여 GraphRAG의 영향력을 보존 |
| Graph hop | 기본 1-hop, 명시적 참조·세부노드 부모 복원·질문어 일치 시 최대 2-hop | 현재 적재된 구조·참조 관계에서 직접 근거를 우선하고, 필요한 경로만 한 단계 더 열어 그래프 폭발을 방지 |
| Graph 예산 | 결과 최대 40개, 2-hop frontier 최대 12개 | Seed가 많아도 재정렬 입력과 지연시간을 제한 |
| 커뮤니티 요약 | 기본 검색에서 제외 | 네 기준서의 정확한 문단 판단에는 원문·지역 그래프가 우선이며, 생성 요약은 인용 근거가 될 수 없음 |
| 답변 출력 | 결론 / 판단 과정 / 근거 | 사용자가 결론과 적용 논리, 원문 근거를 빠르게 검증 가능 |
| 자동 평가 | 구현하지 않음 | 사용자가 직접 평가하기로 결정 |

---

## 3. 전체 시스템 아키텍처

### 색인 파이프라인

```text
K-IFRS HWPX 원문                         PDF 원문
        │                                  │
        ├─ 문서 영역·제목·문단 구조 추출    └─ 페이지 매핑·시각 검증
        ├─ 하위항목·표·각주 보존
        ├─ 명시적 문단 참조 추출
        └─ 원문 JSONL 생성
                     │
                     ▼
              검색용 Chunk 생성
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
OpenAI Embedding             그래프 노드·관계 생성
        │                         │
        └────────────┬────────────┘
                     ▼
                   Neo4j
        ├─ 원문 구조 그래프
        ├─ 의미 그래프
        ├─ Vector Index
        ├─ Full-text Index
        └─ 제약조건·일반 인덱스
```

### 질문 처리 파이프라인

```text
사용자 질문
    │
    ▼
질문 분석
├─ 질문 목적
├─ 기준서·문단번호
├─ 금융상품 개념 후보
└─ 복합 질문의 하위 질문
    │
    ▼
Hybrid 검색
├─ Dense: 임베딩 벡터 검색
└─ Sparse: 전문/키워드 검색
    │
    ▼
RRF 결합 및 Seed 문단 선정
    │
    ▼
Seed 기반 Neo4j Graph 확장
├─ 구조·상하위 항목
├─ 명시적 참조
├─ 정의·적용조건·예외
├─ 적용지침·적용사례
└─ 다른 기준서 연결
    │
    ▼
Seed별 근거 묶음 생성
    │
    ▼
OpenAI 근거 묶음 재정렬
    │
    ▼
답변 전 근거 충분성 검사
    ├─ 부족: 누락된 측면만 최대 1회 재검색
    └─ 충분
         │
         ▼
답변 생성
         │
         ▼
답변 후 인용 검증
         │
         ▼
결론 / 판단 과정 / 근거
```

### 직렬 구조를 선택한 근거

Neo4j의 `HybridCypherRetriever`도 벡터·전문 검색으로 유사 노드를 식별한 뒤 그 노드에서 Cypher 그래프 탐색을 수행한다. 이 구조는 자연어 표현이 그래프의 정확한 엔터티명과 일치하지 않더라도 검색 결과를 안전한 시작점으로 사용할 수 있다.

독립 병렬 검색은 Hybrid와 Graph 양쪽의 후보를 넓게 얻을 수 있지만, 엔터티 연결 실패, 중복 후보, 불필요한 이웃 노드, 비용과 지연시간이 증가한다. 현재 프로젝트처럼 네 개 기준서의 정확한 문단과 조건·예외를 찾는 지역 질의가 중심인 경우에는 직렬 구조가 더 단순하고 추적 가능하다.

참고:

- Neo4j GraphRAG `HybridCypherRetriever`: https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html
- Microsoft GraphRAG Local/Global Search 구분: https://microsoft.github.io/graphrag/query/overview/

---

## 4. 기술 스택과 역할

| 영역 | 기술 | 역할 |
|---|---|---|
| 언어 | Python 3.11+ | 파싱, 적재, 검색, API |
| LLM | OpenAI API | 질문 분석, 의미 관계 후보, 재정렬, 답변·인용 검증 |
| Embedding | OpenAI Embeddings | Chunk 벡터 생성 |
| 데이터베이스 | Neo4j 로컬 → Aura | 그래프·벡터·전문 검색 통합 저장 |
| DB 접근 | Neo4j Python Driver, 명시적 Cypher | 재현 가능한 적재와 제한된 그래프 탐색 |
| RAG 파이프라인 | 프레임워크 없이 직접 구현 | Hybrid 결합, 그래프 확장, 재정렬, 충분성 검사, 인용 검증 |
| 워크플로 | 일반 Python `AccountingQAPipeline` | 질문 분석→검색→확장→재검색→검증 상태 제어 |
| API | FastAPI | 질의 및 근거 응답 |
| 데이터 검증 | Pydantic | JSONL, LLM Structured Output, API 스키마 |
| 테스트 | pytest | 파서·청킹·그래프·검색·답변 회귀 검증 |

**LangChain과 LangGraph는 사용하지 않는다.** 의존성 목록에도 포함하지 않았다. 이 프로젝트가 필요로 하는 동작이 프레임워크의 기본 제공 범위를 벗어나기 때문이다.

- Hybrid 검색 결과를 Seed로 삼아 **기본 1-hop, 조건부 2-hop**으로만 그래프를 확장한다.
- 허용 관계 7종만 통과시키고 `APPEARS_ON`처럼 문서 전체로 확산되는 관계는 제외한다.
- 결과 40개, 2-hop frontier 12개로 탐색 예산을 고정한다.
- 각 근거에 Seed, hop, 관계 경로, 유입 채널을 provenance로 보존한다.

이 제어를 프레임워크 위에 올리면 결국 대부분을 직접 작성한 뒤 그 위에 한 겹을 더 얹는 구조가 된다. 따라서 Neo4j Python Driver와 OpenAI SDK를 직접 호출한다.

워크플로도 마찬가지다. 상태 전이가 `검색 → 충분성 검사 → (부족하고 누락 측면이 명확할 때만) 1회 재검색`뿐이라 일반 `for` 반복문으로 충분하다. 스트리밍·중단·재개·장기 실행 상태가 실제로 필요해지면 같은 상태 전이를 LangGraph로 옮길 수 있지만, 현재 기능을 위해 불필요하게 의존성을 추가하지 않는다.

LLM이 임의의 Cypher를 생성하여 전체 DB를 탐색하게 하지 않는다. 운영 검색은 관계 유형, 방향, 깊이, 최대 노드 수가 정해진 Cypher 템플릿을 사용한다.

---

## 5. 디렉터리 구조

아래는 설계 초기에 그린 목표 구조다. 실제 구현에서는 `docs/`·`cypher/`·`prompts/`를 별도로 두지 않고 `NEO4J_SCHEMA.md`를 루트에, DDL을 `db/`에, 프롬프트를 `config/*.yaml` 안에 통합했다. 현재 실제 구조는 이 문서 하단의 구현 단계와 저장소 트리를 따른다.

```text
accounting_rag/
├── PROJECT_STRUCTURE.md
├── PROJECT_WORKFLOW.html
├── README.md
├── pyproject.toml
├── .env
├── .env.example
├── config/
│   ├── retrieval.yaml            # top-k, RRF, hop, 후보 예산
│   ├── graph_schema.yaml         # 허용 노드·관계·속성
│   └── prompts.yaml              # 질문 분석·재정렬·검증 프롬프트
├── data/
│   ├── raw/
│   │   └── standards/
│   ├── processed/
│   │   ├── paragraphs.jsonl
│   │   ├── blocks.jsonl
│   │   ├── tables.jsonl
│   │   ├── footnotes.jsonl
│   │   ├── references.jsonl
│   │   ├── unresolved_references.jsonl
│   │   ├── unparsed_reference_candidates.jsonl
│   │   ├── document_manifest.json
│   │   ├── graph_candidates.jsonl
│   │   ├── graph_approved.jsonl
│   │   └── QUALITY_REPORT.md
│   └── chunks/
│       ├── chunks.jsonl
│       └── CHUNK_QUALITY_REPORT.md
├── docs/
│   ├── NEO4J_SCHEMA.md
│   └── ANSWER_POLICY.md
├── cypher/
│   ├── schema.cypher
│   ├── ingest.cypher
│   └── validation.cypher
├── prompts/
│   ├── question_analysis.md
│   ├── graph_extraction.md
│   ├── evidence_reranking.md
│   ├── answer_generation.md
│   └── citation_verification.md
├── src/
│   └── accounting_rag/
│       ├── api/
│       │   ├── app.py                 # FastAPI /health, /v1/ask
│       │   ├── dependencies.py        # 실제 파이프라인 조립·수명주기
│       │   └── schemas.py             # 요청·응답 Pydantic 스키마
│       ├── ingestion/
│       │   ├── hwpx_parser.py
│       │   ├── dataset_builder.py
│       │   ├── chunk_paragraphs.py
│       │   ├── extract_graph_candidates.py
│       │   └── ingest_neo4j.py
│       ├── retrieval/
│       │   ├── embeddings.py
│       │   ├── hybrid.py
│       │   ├── graph_expansion.py
│       │   ├── pipeline.py
│       │   └── reranker.py
│       ├── query/
│       │   └── analysis.py             # 검색용 기준서·문단·개념·하위 질문 분석
│       ├── generation/
│       │   ├── answer.py               # 일반 근거 답변
│       │   ├── sufficiency.py          # 충분성 검사
│       │   └── citation_verifier.py    # 답변 후 인용 검증
│       └── qa_pipeline.py              # 분석→검색→검사→답변 통합 흐름
├── scripts/
│   ├── parse_all_standards.py
│   ├── validate_processed_dataset.py
│   ├── build_chunks.py
│   ├── build_graph.py
│   ├── build_indexes.py
│   └── ask.py                          # 전체 질의응답 CLI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
└── infra/
    └── neo4j/
        └── docker-compose.yml
```

---

## 6. 원문 구조와 청킹 정책

### 원문 구조

```text
Standard
└─ Zone
   └─ Chapter/Section
      └─ Paragraph
         ├─ Subparagraph
         │  └─ 더 하위 Subparagraph
         ├─ Table
         └─ Footnote
```

공식 문단번호가 가장 강한 경계다. 문단 내부의 `⑴`, `㈎`, `(1)`, `가.` 등은 소분류로 인식한다. 원문 계층과 검색 단위는 분리한다.

### 모든 내용 반영 정책

- 공식 본문: 기본 검색의 최우선 근거
- 공식 부록·적용지침·용어 정의: 본문 다음 우선순위
- 적용사례·실무적용지침: 계산과 사실관계 설명의 보조 근거
- 결론도출근거·소수의견: 사용자가 요구하거나 해석 배경이 필요한 경우에만 사용
- 표지·저작권·목차·빈 문단·단순 편집정보: 파싱하여 보존하지만 기본 검색에서 제외
- 삭제 표시·시행일·경과규정: 버전 판단을 위해 별도 속성으로 보존

### Chunk 생성 원칙

1. 원본 `Paragraph`와 `Subparagraph`는 수정하지 않는다.
2. 기본 Chunk는 공식 문단 하나다.
3. 문단이 길면 하위 소분류 경계에서 분할한다.
4. 문단이 짧더라도 정의나 독립 규칙이면 단독 Chunk로 유지한다.
5. 짧은 문단도 다른 공식 문단과 자동 병합하지 않는다.
6. 표는 구조화 데이터와 검색용 직렬화 텍스트를 모두 만든다.
7. 각주는 관련 본문 Chunk에 문맥으로 포함하고 별도 원문 노드도 유지한다.
8. 기계적인 고정 overlap 대신 `NEXT`와 계층 관계로 주변 문맥을 확장한다.
9. 실제 길이 분포를 측정한 뒤 분할 임계값을 확정한다.

실제 3,904개 문단의 모든 Paragraph·Subparagraph·continuation Block을 합산해 측정한 결과는 다음과 같다.

- 문단 원문 길이 중앙값: 250자
- 95백분위: 656자
- 99백분위: 1,224자
- 1,200자 초과 문단: 40개
- 소분류: 2,125개
- continuation이 있는 문단: 447개
- 가장 긴 문단: 4,554자
- 가장 긴 표: 6,653자

이 분포를 근거로 짧은 문단은 자동 병합하지 않는다. 문단 인용 경계를 유지하고 검색 결과를 확인한 뒤에만 병합 필요성을 다시 검토한다. 현재 설정은 문단 목표/최대 900/1,200자, 표 목표/최대 1,200/1,800자이며, 긴 문단은 Block·Subparagraph 경계, 표는 행 경계에서 분할한다.

### Chunk 필수 메타데이터

```json
{
  "chunk_id": "KIFRS1109-5.5.17-C01",
  "text": "검색에 사용할 텍스트",
  "standard_id": "1109",
  "zone": "standard_body",
  "section_path": ["손상", "기대신용손실"],
  "source_paragraph_ids": ["KIFRS1109-5.5.17"],
  "source_subparagraph_ids": [],
  "table_ids": [],
  "footnote_ids": [],
  "pdf_page_start": 120,
  "pdf_page_end": 121,
  "chunk_order": 1,
  "search_priority": 100,
  "version": "원문 판본"
}
```

`Chunk`는 검색을 위한 파생 데이터이고 법적·회계적 원문 단위가 아니다. 따라서 반드시 `DERIVED_FROM` 관계로 원문 문단 또는 소분류에 연결한다.

---

## 7. 필요한 세 종류의 스키마

### 7.1 원문/Chunk 데이터 스키마

JSONL 각 행의 필드, 자료형, 필수 여부와 ID 규칙을 Pydantic으로 검증한다. 파싱 데이터와 Chunk 데이터는 분리한다.

### 7.2 Neo4j 그래프 스키마

노드 라벨, 관계 유형, 속성, 고유키, 탐색 방향과 허용 hop을 명시한다. Neo4j가 스키마 없이도 데이터를 저장할 수 있다는 사실과 별개로, 재현 가능한 GraphRAG를 위해 애플리케이션 스키마를 먼저 확정한다.

### 7.3 LLM Structured Output 스키마

OpenAI가 생성하는 질문 분석, 의미 관계 후보, 재정렬 결과, 근거 충분성 결과와 답변을 JSON Schema로 제한한다. 자유형 텍스트를 다시 정규식으로 해석하지 않는다.

---

## 8. Neo4j 그래프 스키마

### 8.1 원문·구조 노드

| 라벨 | 의미 | 주요 속성 |
|---|---|---|
| `Standard` | 제1032·1039·1107·1109호 | `standard_id`, `title`, `version` |
| `Zone` | 본문·부록·사례·결론도출근거 등 | `zone_id`, `type`, `search_priority` |
| `Section` | 장·절·소제목 | `section_id`, `title`, `level`, `order` |
| `Paragraph` | 공식 번호가 있는 문단 | `paragraph_id`, `number`, `text`, `order` |
| `Subparagraph` | 문단 내부의 번호 있는 소분류 | `subparagraph_id`, `marker`, `text`, `level`, `order` |
| `Table` | 행·열·병합을 보존한 표 | `table_id`, `title`, `serialized_text` |
| `Footnote` | 각주 표식과 본문 | `footnote_id`, `marker`, `text` |
| `Chunk` | 검색용 텍스트 | `chunk_id`, `text`, `embedding`, `search_priority` |
| `ExternalStandard` | 원문 미적재 외부 기준서 | `standard_id`, `title`, `loaded=false` |

### 8.2 의미 노드

| 라벨 | 의미 | 필수 추적정보 |
|---|---|---|
| `Concept` | 금융자산, 상각후원가, ECL 등 | 대표명, 공식 정의, 승인된 동의어, 정의 출처 |

의미 노드는 `Concept` 하나만 사용한다. 네 기준서의 공식 용어 정의(제1107·1109호 정의표, 제1032호 문단 11, 제1039호 문단 9)에서 결정적으로 생성하므로 LLM 추론이 개입하지 않는다. 의미 노드와 관계에는 `confidence`, `source_id`, `source_text_span`, `extractor_model`, `extractor_version`, `review_status`를 저장한다.

`Rule`·`Condition`·`Exception`·`Example`은 설계 검토 후 최종 범위에서 제외했다(위 "의미 규칙 KG — 시도 후 제거" 참고).

### 8.3 주요 관계

```text
(:Standard)-[:CONTAINS]->(:Zone)
(:Zone)-[:CONTAINS]->(:Section)
(:Section)-[:CONTAINS]->(:Paragraph)
(:Paragraph)-[:CONTAINS]->(:Subparagraph)
(:Subparagraph)-[:CONTAINS]->(:Subparagraph)

(:Paragraph)-[:NEXT]->(:Paragraph)
(:Subparagraph)-[:NEXT]->(:Subparagraph)
(:Paragraph)-[:HAS_TABLE]->(:Table)
(:Paragraph)-[:HAS_FOOTNOTE]->(:Footnote)
(:Chunk)-[:DERIVED_FROM]->(:Paragraph|Subparagraph|Table)

(:Paragraph)-[:REFERS_TO]->(:Paragraph)
(:Paragraph)-[:REFERS_TO]->(:ExternalStandard)
(:Paragraph|Block|Table)-[:MENTIONS]->(:Concept)
```

### 8.4 관계 생성 신뢰도

- A등급: HWPX 계층, 문단 순서, 명시적 기준서·문단번호 및 공식 용어 정의에서 규칙 기반으로 생성한다. 공식 `Concept`와 정확 문자열 `MENTIONS`도 포함하며 기본 검색에 자동 사용한다.
- B등급: OpenAI가 후보를 만들고 실제 근거 문구·문단 존재·관계 방향을 검증한 의미 관계다. 승인 후 사용한다.
- C등급: 의미상 관련 가능성은 있지만 검토되지 않은 후보다. 검색에서 제외하고 검토 파일에 저장한다.

OpenAI가 생성한 후보는 바로 확정 관계로 병합하지 않는다. 동일 개념 후보도 동의어 사전과 근거를 확인한 후 병합한다.

### 8.5 범위 참조

`문단 3.2.3~3.2.9`, `3.2.3-3.2.9`, `102D∼102I`, `BC220E–BC220G`처럼 범위가 명시되면 기호를 내부적으로 정규화한다. 대상 기준서의 실제 문단 순서표에서 시작과 끝 사이에 존재하는 문단만 펼쳐 각각 `REFERS_TO`를 생성한다.

원문 표현, 그룹 ID, 시작·끝 번호와 해석 상태를 관계 속성으로 보존한다. 존재하지 않는 가상 문단은 만들지 않는다. 시작 또는 끝을 확정할 수 없으면 `unresolved_references.jsonl`에 남기고 운영 그래프에서 제외한다.

### 8.6 제약조건과 인덱스

- 모든 주요 노드 ID에 UNIQUE 제약조건
- `Paragraph.number`, `standard_id`, `zone`, `order`에 조회 인덱스
- `Chunk.embedding`에 3,072차원 Vector Index `chunk_embedding_vector`
- `Chunk.text`와 필요한 제목·문단 필드에 `cjk` analyzer 기반 Full-text Index. 한국어 복합어 검색 실측 결과 `standard-no-stop-words`보다 적합하여 마이그레이션으로 변경
- 적재 스크립트는 재실행해도 중복 노드와 관계를 만들지 않도록 `MERGE`와 안정적인 ID 사용

---

## 9. 질문 분석

질문 분석은 DB의 키워드만으로 질문을 분류하는 과정이 아니다.

1. 규칙 기반 추출로 기준서 번호와 문단번호를 식별한다.
2. Neo4j의 `Concept.name`, `aliases`, 절 제목과 규칙명을 검색해 엔터티 후보를 연결한다.
3. OpenAI Structured Output으로 질문 목적과 하위 질문을 판단한다.
4. 엔터티 연결이 실패해도 Hybrid 검색은 항상 수행할 수 있어야 한다.

```json
{
  "requested_standard_ids": ["1109"],
  "requested_paragraphs": [],
  "concepts": ["금융자산 제거"],
  "subquestions": ["제거 조건", "제거하지 못한 경우의 처리"],
  "search_query": "K-IFRS 1109 금융자산 제거 조건과 후속 처리"
}
```

질문 분석기는 Hybrid와 Graph 중 하나를 배타적으로 선택하는 라우터가 아니다. 기본 경로는 항상 Hybrid → Graph 확장이다. 정확한 문단번호가 있는 경우에만 해당 노드를 직접 Seed로 사용하는 최적화를 둔다.

현재 `query/analysis.py`는 기준서·문단번호를 결정적으로 추출한 뒤 OpenAI Responses strict JSON Schema로 검색 질의, 개념과 하위 질문을 구조화한다. 객관식·주관식·계산형을 분류하지 않는다. 모델이 사용자가 명시한 기준서·문단을 바꾸거나 알 수 없는 기준서를 추가하면 원문 질문을 보존하는 fallback으로 돌아간다. 분석 결과는 Hybrid 검색 조건으로만 사용한다.

---

## 10. Hybrid 검색과 RRF

### Dense 검색

OpenAI 임베딩으로 질문을 벡터화하고 Neo4j Vector Index에서 의미가 유사한 `Chunk`를 검색한다.

### Sparse 검색

Neo4j `chunk_fulltext`의 `cjk` analyzer로 한국어 회계용어를 검색하고, 기준서 번호와 문단번호는 원문 필드와 함께 조회한다. `기대신용손실` probe가 유효한 결과를 반환하는지 검증기에 포함한다.

### RRF

Dense와 Sparse의 원점수는 직접 비교하지 않는다. 각 검색 결과의 순위를 Reciprocal Rank Fusion으로 결합한다.

초기 후보 예산:

- Dense 상위 20개
- Sparse 상위 20개
- RRF 결합 후 Seed 상위 10개

숫자는 설정 파일에 두고 실제 문단 길이와 검색 로그를 검토한 후 조정한다.

---

## 11. Graph 확장과 hop 정책

`hop`은 그래프 관계 Edge 하나를 건너가는 횟수다. 검색용 `Chunk`를 원문 노드로 변환하는 `DERIVED_FROM` 이동은 도메인 의미 hop 계산에서 제외하고, 원문 `Paragraph` 또는 `Subparagraph`에서 탐색을 시작한다.

### 현재 구현된 확장 정책

- 기본 1-hop, 조건부 최대 2-hop
- 2-hop은 명시적 `REFERS_TO`, `MENTIONS`, 세부 노드에서 부모 문단을 복원하는 경로, 또는 질문어가 겹치는 후보에만 허용
- 허용 관계: `REFERS_TO`, `MENTIONS`, `CONTAINS`, `HAS_BLOCK`, `HAS_TABLE`, `HAS_FOOTNOTE`, `NEXT`
- 의미 경로는 `검색 원문 → Concept → 공식 정의 출처`로 제한한다. 모든 등장 문단으로 역확장하지 않아 공정가치·금융상품 같은 고빈도 개념의 그래프 폭발을 막는다.
- `Concept`는 탐색용 중간 노드이며 최종 답변 근거 후보에서는 제외한다. 인용 가능한 공식 정의표·정의 블록만 재정렬기에 전달한다.
- `APPEARS_ON`과 문서 전체로 확산될 수 있는 관계는 확장에서 제외
- 결과 최대 40개, 2-hop frontier 최대 12개
- 범위 참조는 각 대상에 직접 `REFERS_TO`가 있으므로 `NEXT`를 반복 순회하지 않음

### 그래프 폭발 방지

- 방문 노드 집합으로 순환 방지
- 관계 유형과 방향 화이트리스트
- 질문별 결과 최대 40개와 2-hop frontier 최대 12개
- 같은 기준서와 같은 영역 우선
- C등급 의미 관계 제외
- 검색 이유와 전체 경로 기록

---

## 12. 근거 묶음과 OpenAI 재정렬

Graph 확장 결과는 개별 문단 목록이 아니라 Seed별 근거 묶음으로 만든다.

```json
{
  "bundle_id": "BUNDLE-001",
  "seed_chunk_ids": ["KIFRS1109-3.2.6-C01"],
  "seed_paragraph_ids": ["KIFRS1109-3.2.6"],
  "supporting_paragraph_ids": ["KIFRS1109-3.2.7", "KIFRS1109-3.2.9"],
  "conditions": ["위험과 보상의 이전 여부"],
  "exceptions": ["계속관여"],
  "graph_paths": [
    "KIFRS1109-3.2.6 -REFERS_TO-> KIFRS1109-3.2.9"
  ],
  "retrieval_reasons": ["dense", "fulltext", "graph_reference"]
}
```

OpenAI 재정렬기는 통합 후보를 최대 48개까지 입력받을 수 있고, 기본 상위 10개를 반환한다. 실제 통합 검증에서는 Hybrid Seed 10개와 그래프 근거 40개로부터 중복을 제거한 재정렬 후보 15개를 구성해 결과 10개를 반환했다. 정확한 문단 원문 조회는 후속 직접 조회 라우팅에서 재정렬을 생략할 수 있다.

평가 기준:

- 질문과의 직접 관련성 40%
- 기준서 본문 우선도 20%
- 직접 적용 근거성 30%
- 명시적 참조·그래프 경로 연결성 10%

재정렬 결과는 OpenAI Responses API의 JSON Schema Structured Output으로 후보별 0~5점, 선택 ID와 한국어 이유를 받는다. 응답 누락·중복·알 수 없는 ID·형식 오류 또는 API 호출 실패 시 입력 후보의 결정적 순서를 유지하는 fallback으로 검색 흐름을 계속한다.

---

## 13. 근거 충분성 검사

### 답변 생성 전

다음 조건을 모두 검사한다.

1. 질문에 직접 답할 수 있는 공식 본문 또는 공식 부록 문단이 최소 하나 존재한다.
2. 복합 질문의 각 하위 질문에 하나 이상의 근거가 연결된다.
3. 핵심 문단의 직접 `REFERS_TO` 대상 중 판단에 필요한 문단이 포함된다.
4. 적용사례나 결론도출근거만 있고 규범적 본문이 없는 경우 이를 충분한 규정 근거로 보지 않는다.
5. 외부 기준서가 필요하지만 원문이 적재되지 않았다면 범위 제한을 표시한다.

부족하면 `missing_aspects`를 생성하여 해당 부분만 질문을 확장하고 최대 1회 재검색한다.

```json
{
  "sufficient": false,
  "covered_aspects": ["금융자산 제거 조건"],
  "missing_aspects": ["제거하지 못한 경우의 회계처리"],
  "required_relationships": ["REFERS_TO", "HAS_EXCEPTION"],
  "retry_query": "금융자산 제거조건 미충족 계속인식 회계처리"
}
```

현재 구현은 먼저 후보 수·고유 ID·본문·인용·재정렬 점수·직접 근거 점수를 결정적으로 확인하고, 통과한 근거만 OpenAI 의미 충분성 검사에 전달한다. 의미 검사는 주장별 coverage, 지원 근거 ID, 누락 측면, 충분 여부와 confidence를 strict JSON Schema로 반환한다. API 오류, 알 수 없는 근거 ID, 낮은 confidence 또는 남은 누락 측면은 보수적으로 불충분 처리한다. `qa_pipeline.py`는 `missing_aspects`가 있을 때만 보완 검색어를 만들며 재검색은 최대 1회로 고정한다.

### 답변 생성 후

- 결론과 판단 과정의 주요 주장마다 인용 문단이 존재하는지 확인한다.
- 인용 문단이 실제로 해당 주장을 지지하는지 검사한다.
- 존재하지 않는 기준서·문단번호를 차단한다.
- 근거 밖의 문장은 제거하거나 불확실성을 표시한다.

모든 질문은 같은 답변 생성기를 사용한다. 보기가 있으면 결론에 정답을, 계산이 필요하면 판단 과정에 계산을 포함하도록 질문 원문이 모델에 그대로 전달된다. 각 결론·판단 문장에는 제공된 evidence ID가 필요하며 근거 ID 위조나 지원되지 않는 주장은 인용 검증에서 차단한다.

실제 ECL 질문의 전체 실행에서는 충분성 confidence 0.98과 검증된 인용 4개로 `answered` 결과를 확인했다.

---

## 14. 사용자 답변 형식

사용자에게는 다음 세 구역만 보여준다.

```text
[결론]
질문에 대한 직접적인 답변

[판단 과정]
적용 조건, 순서, 예외와 필요한 계산

[근거]
K-IFRS 제1109호 문단 3.2.6
K-IFRS 제1109호 문단 3.2.9
```

PDF 페이지와 그래프 경로는 근거 패널에서 확인할 수 있게 하되, 본문 출력의 최상위 제목은 늘 위 세 개로 유지한다. 기준서 문구와 모델의 해석을 구분하며, 범위 밖 질문은 현재 적재된 네 기준서만으로 확인하기 어렵다고 명시한다.

---

## 15. 환경변수와 설정

```env
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL=gpt-5.6-terra
OPENAI_RERANK_MODEL=gpt-5.6-terra
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

`.env`에는 비밀정보와 환경별 연결값·모델명만 저장하고 Git에 커밋하지 않는다.

배포 전용 선택 환경변수(미설정 시 로컬 동작 불변):

```env
# 분리 배포된 프론트엔드 origin 허용 목록(쉼표 구분)
CORS_ALLOW_ORIGINS=https://your-site.pages.dev
# IP당 시간당 /v1/ask 요청 제한. 0 또는 미설정이면 무제한
ASK_RATE_LIMIT_PER_HOUR=10
# 리버스 프록시 뒤에서만 X-Forwarded-For를 신뢰
TRUST_PROXY_HEADERS=1
```

`config/retrieval.yaml`에는 다음을 저장한다.

- Dense/Sparse top-k
- RRF 상수와 Seed 수
- 관계별 허용 방향과 hop
- 후보 노드·근거 묶음·컨텍스트 토큰 예산
- 재검색 횟수

`config/prompts.yaml` 또는 `prompts/`에는 질문 분석, 관계 후보 추출, 재정렬, 충분성 검사, 답변과 인용 검증 프롬프트를 버전 관리한다.

---

## 16. 구현 단계와 완료 조건

### Phase 1. 원문 확보와 파서 시제품 — 완료

- 네 기준서 HWPX·PDF 확보
- 1109호 대표 문단의 문단·하위항목·표·각주·페이지 대조
- 사용자 원문 검수 완료

### Phase 2. 네 기준서 전체 구조화·청킹 — 완료

- 문단 3,904개
- 원문 블록 8,533개
- 표 362개
- 각주 227개
- 명시적 참조 7,450개
- 숫자형 문단 참조 미파싱 후보 0개
- 대상 미확정 참조 190개 별도 보존
- 검색용 Chunk 4,379개 생성
- 문단 Chunk 3,864개, 긴 문단 분할 Chunk 105개, 표 Chunk 410개
- 내용이 있는 원문 문단 3,904개, 소분류 2,125개, 내용 Block 6,462개, 표 362개, 소속 각주 215개 역추적 검증
- Chunk ID 중복·빈 Chunk·길이 제한 위반 0개
- PDF 1,671쪽을 기준으로 문단 3,904/3,904, 검색 대상 문단 3,891/3,891 페이지 매핑
- 검색 대상 Chunk 4,144/4,144개에 PDF 페이지 부여(미매핑 0개)
- 매핑 방식: exact 3,900개, fuzzy 1개, inferred_next 3개
- 저신뢰도 101개와 복수 후보 216개를 포함한 검토 목록 287개 별도 생성
- 표 341/362개 매핑. 미매핑 표는 대부분 실질 내용이 없는 orphan이며 검색 대상 표 Chunk는 모두 페이지 보유
- 각주 227/227개 매핑
- 구현 시점 기준 전체 테스트 123개 통과(과거 실행 기록)
- PDF 42·119·142·317·318쪽과 제1032호 85쪽 시각 표본 대조 완료
- 독립 감사에서 IE142의 317~318쪽, `KIFRS1032-T-0049`의 84~85쪽을 수정하고 16C·35M·7.2.12의 앞쪽 오탐을 제거했다. B3.2.17 다음 절 제목은 문단 경계에서 분리하고 anchor 후보 페이지는 감사 추적용으로 보존한다.

### Phase 3. Neo4j 스키마와 구조·참조 적재 — 완료

- `NEO4J_SCHEMA.md`, `config/graph_schema.yaml`, `db/schema.cypher`와 `db/migrations/002_rebuild_chunk_fulltext_cjk.cypher` 작성·적용
- `scripts/load_neo4j.py`, `scripts/validate_neo4j.py`로 재실행 가능한 로컬 적재와 검증 구현
- 구조 노드와 A등급 확정 관계 및 명시적 범위 참조 적재
- 노드: Standard 4, Zone 20, Section 713, Paragraph 3,904, Subparagraph 2,125, Block 8,533, Table 362, Footnote 227, PdfPage 1,671, Chunk 4,379, ExternalStandard 35
- 관계 74,409개: APPEARS_ON 19,968, CONTAINS 8,433, DERIVED_FROM 14,087, HAS_BLOCK 6,674, HAS_FOOTNOTE 442, HAS_TABLE 574, NEXT 11,435, REFERS_TO 12,796
- 제약조건·조회 인덱스와 `chunk_fulltext`, `chunk_embedding_vector` 생성 및 ONLINE 확인. `chunk_fulltext`는 한국어 실측 후 `cjk` analyzer로 재구축
- Neo4j 검증기가 Full-text analyzer=`cjk`, `기대신용손실` probe와 인덱스 ONLINE 상태를 확인하며 `valid=true`
- 공식 정의 기반 `Concept` 45개와 승인된 `MENTIONS` 17,712개 생성·적재. 그중 47개는 정의 출처 관계이며 `concept_fulltext` 인덱스를 생성
- 동일 공식 용어는 하나의 공유 Concept로 병합하고, 원문에 명시된 장·단기 명칭만 alias로 승인. 임의의 AC·ECL 약어는 만들지 않음

구조·개념 적재 완료 조건: 대표 문단에서 원문 계층, 표·각주, 참조 범위와 `문단 → Concept → 공식 정의` 경로를 Cypher로 재현하고 중복 없이 재적재할 수 있다. `scripts/validate_semantic_kg.py`가 개수·출처·승인 상태·ECL 경로·인덱스를 검증한다.

### Phase 4. 임베딩과 Hybrid 검색 핵심 — 완료

- 검색 대상 Chunk 4,144개의 `text-embedding-3-large` 3,072차원 임베딩 생성 및 Neo4j `Chunk.embedding` 적재
- `data/embeddings/chunk_embeddings.jsonl`, `embedding_manifest.json`, `EMBEDDING_QUALITY_REPORT.md` 생성
- `scripts/build_embeddings.py`, `scripts/load_embeddings_neo4j.py`, `scripts/validate_embeddings.py`로 캐시 생성·적재·검증 자동화
- `src/accounting_rag/retrieval/hybrid.py`에서 Dense `chunk_embedding_vector` + Sparse `chunk_fulltext(cjk)` 검색 구현
- `config/retrieval.yaml`의 Dense/Sparse 후보 수, 가중치와 RRF k를 사용하는 weighted RRF 결합
- `scripts/query_hybrid.py` CLI와 기준서·영역 메타데이터 필터 구현
- 정확한 문단 직접 조회 라우팅과 운영 검색 로그는 후속 파이프라인에서 완성
- 구현 시점 기준 전체 테스트 123개 통과(과거 실행 기록)

핵심 완료 조건: 질문별 Dense/Sparse 후보의 원점수·순위·유입 채널과 weighted RRF Seed가 반환된다. 운영 로그 영속화는 답변 파이프라인에서 추가한다.

### Phase 5. GraphRAG와 재정렬 — 완료

- `graph_expansion.py`: Hybrid Seed를 `DERIVED_FROM` 원문 노드로 복원하고 허용 관계만 기본 1-hop·조건부 최대 2-hop 탐색
- 결과 최대 40개, 2-hop frontier 최대 12개로 제한하고 방문 집합으로 순환 방지
- 각 근거에 Seed Chunk, hop, 관계 유형, 경로, provenance와 graph score 보존
- `pipeline.py`: Hybrid → Graph → 중복 없는 후보 통합 → rerank 순서를 단일 흐름으로 구현
- `reranker.py`: OpenAI Responses JSON Schema Structured Output, 네 평가 가중치와 결정적 fallback 구현
- `scripts/query_retrieval.py`: 기준서·영역 필터와 결과 수를 지원하는 통합 CLI 구현
- 실제 검증: Seed 10개, 그래프 근거 40개, 재정렬 후보 15개, 결과 10개. B5.5.51·B5.5.52 등 직접 적용지침이 최상위로 개선
- 구현 시점 기준 전체 테스트 123개 통과(과거 실행 기록)

완료 조건: 선택된 각 결과에서 Hybrid 유입 채널 또는 Seed, 그래프 경로, hop, 관계 유형과 최종 원문을 역추적할 수 있다.

### Phase 6. 근거 기반 답변 — 완료

- 규칙 추출 + OpenAI strict Structured Output 질문 분석 및 안전한 fallback
- 분석 결과를 Hybrid 검색 조건에만 연결하고 답변은 단일 생성 경로 사용
- 답변 전 결정적 조건과 의미 판단을 결합한 근거 충분성 검사
- `missing_aspects`가 존재할 때만 최대 1회 누락 측면 재검색
- 모든 질문에 동일한 `결론 / 판단 과정 / 근거` 생성
- 답변 후 evidence ID, citation, 실제 후보 존재와 문장별 인용 무결성 검증
- API·구조화 응답 오류 시 추측 답변을 노출하지 않는 보수적 fallback
- 실제 ECL 질문 결과: `answered`, 충분성 confidence 0.98, 검증된 인용 4개

완료 조건: 모든 주요 주장에 실제 적재 문단이 연결되고, 근거 부족·범위 밖 질문을 명시적으로 처리한다.

### Phase 7. API와 UI — 완료

- FastAPI 앱과 `/health`, `/v1/ask` 구현 완료
- 질문·선택적 기준서/영역·top-k·debug 요청 검증 완료
- 검증된 결론·판단 과정·근거 응답과 안전한 422/500/503 오류 응답 완료
- debug 응답에 검색 건수, 재검색 시도, 충분성 confidence·누락 측면, 질문 분석과 인용 검증 제공
- 질문 입력, 네 기준서·11개 문서 영역·근거 수 필터, 로딩·오류·진단 상태를 보여주는 반응형 사용자 UI 완료
- 결론·판단과정·근거 원문과 함께 PDF 페이지, Hybrid/Graph 출처, graph hop 및 관계 경로 표시 완료

완료 조건: 사용자가 질문부터 답변, 인용 원문과 그래프 경로 확인까지 한 화면에서 수행할 수 있다.

### Phase 8. 프로젝트 마무리 — 로컬 완성

- `.env`와 로컬 원본·파생 데이터를 공개 저장소에서 제외
- README, 데이터 명세, 운영 범위와 제한사항 문서화
- 객관식·계산형 전용 라우팅 없이 단일 답변 경로로 확정
- 추가 자동 테스트·CI·Docker·Aura·운영 관찰 기능은 최종 범위에서 제외

완료 조건: 로컬 환경에서 현재 기능을 사용할 수 있고, 공개 저장소에는 코드·스키마·설정 예시·문서만 포함된다.

### Phase 9. 무료 클라우드 배포 — 완료

로컬 완성본을 전액 무료 구성으로 공개 배포했다.

| 계층 | 서비스 | 비고 |
|---|---|---|
| 데이터베이스 | Neo4j AuraDB Free | 3일 미사용 시 일시정지, 로컬 데이터가 원본 |
| 백엔드 | Render 무료 웹 서비스 (Docker) | 15분 유휴 시 슬립, 첫 요청 지연 발생 |
| 프론트엔드 | Cloudflare Pages | 정적 자산 직접 업로드 |

백엔드는 처음에 Hugging Face Spaces를 검토했으나, Docker Space가 유료 구독 전용으로 바뀌어 Render로 변경했다.

배포를 위해 추가한 사항:

- `CORS_ALLOW_ORIGINS`로 프론트엔드 origin만 허용
- `ASK_RATE_LIMIT_PER_HOUR`로 IP당 시간당 요청 제한(OpenAI 비용 보호)
- `TRUST_PROXY_HEADERS`로 리버스 프록시 뒤에서만 `X-Forwarded-For` 신뢰
- `api/static/config.js`로 프론트엔드 API 주소 분리
- `Dockerfile`(Render `PORT` 환경변수 대응)과 `scripts/build_frontend_dist.py`
- 위 환경변수는 모두 선택 항목이며 미설정 시 로컬 동작이 바뀌지 않는다

Aura 적재 결과는 로컬과 동일하다. 노드 22,018개, 관계 92,121개, 검색 대상 Chunk 4,144개 전체 임베딩 보유, `chunk_embedding_vector`와 `chunk_fulltext(cjk)` 모두 ONLINE으로 검증했다.

### Phase 10. 비동기 답변과 이미지 첨부 — 완료

답변 생성에 30~60초가 걸려 브라우저 연결이 끊기면 결과를 잃는 문제가 있었다. 요청을 작업 단위로 분리했다.

- `POST /v1/jobs`가 즉시 `job_id`를 반환하고 백그라운드 스레드가 파이프라인을 수행한다.
- 클라이언트는 `GET /v1/jobs/{job_id}`로 `pending → complete | error`를 폴링한다.
- 작업은 프로세스 메모리에 10분간만 보관하며 만료분은 자동 정리한다.
- `DELETE /v1/jobs/{job_id}`로 조기 정리할 수 있다.
- 답변 대기 중 새로고침해도 폴링을 이어간다.

이미지 첨부도 추가했다. 첨부 이미지는 OpenAI 비전 모델로 문구·숫자·표 구조를 텍스트로 옮긴 뒤 질문에 덧붙여 기존 검색 경로를 그대로 탄다. 이미지는 서버에 저장하지 않고 OpenAI에도 `store=False`로 전달한다. PNG·JPEG·WEBP·GIF, 최대 4장, 장당 5MB로 제한하며 서버에서 base64 유효성과 크기를 다시 검증한다.

기존 `POST /v1/ask` 동기 엔드포인트도 유지한다.

### 의미 규칙 KG — 시도 후 제거

`Rule`·`Condition`·`Exception`·`Example` 후보 추출기와 사람 검토 UI를 한 차례 구현했으나 최종적으로 제거했다. 후보를 실제로 생성·승인하지 않아 검색에 사용된 적이 없고, 승인 절차를 운영할 계획이 없기 때문이다. 관련 노드·관계 정의도 `config/graph_schema.yaml`에서 함께 삭제했다.

검색은 A등급 사실 그래프(문단 구조, 명시적 참조, 공식 정의 기반 Concept)만 사용한다.

---

## 17. 최종 상태

로컬 기능 범위와 무료 클라우드 배포가 모두 완료됐다. 추가 자동 테스트, 의미 규칙 KG, 운영 로그, 피드백 저장은 필수 작업이 아니라 향후 선택 사항이다.

### 알려진 한계

- **정의를 직접 묻는 질문의 검색 편향**: 부록 정의표는 여러 용어를 한 Chunk에 담고 있어, 해당 용어를 자주 언급하는 본문·결론도출근거 문단에 순위가 밀릴 수 있다. 실측에서 "유효이자율의 정의"를 물었을 때 정의표(`KIFRS1109-T-0014`)가 상위 10개에 들지 못해 근거 부족으로 처리됐다. 정의 질문에 정의 영역 가산점을 주거나 정의표를 용어 단위로 재청킹하면 개선할 수 있다.
- **좁은 근거 질문**: 충분성 검사가 자격 근거 2개 이상을 요구하므로 단일 문단에만 존재하는 내용은 기각될 수 있다. 실측한 정의·문단 지정 질문은 모두 통과했으나 구조적 가능성은 남아 있다. `config/answering.yaml`에서 임계값을 조정할 수 있다.
- **콜드 스타트**: Render 무료 플랜은 15분 유휴 후 슬립하여 첫 요청이 최대 1분가량 지연된다.

GitHub 공개 시에는 `.gitignore`에 따라 기준서 원본, 파싱 원문, Chunk, 임베딩, 의미 KG 산출물과 `.env`를 제외하고 코드·스키마·설정 예시·문서만 게시한다. 포트폴리오 저장소에는 사용자 결정에 따라 웹 UI 구현을 제외하고 RAG 구현만 포함한다.
