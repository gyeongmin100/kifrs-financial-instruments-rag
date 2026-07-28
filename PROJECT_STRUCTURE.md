# Accounting RAG 프로젝트 구조

이 문서는 시스템이 **지금 어떻게 동작하는지**와 **왜 그렇게 되었는지**를 담는다. 코드를 처음 보는 사람이 이 문서만으로 전체 흐름과 각 선택의 이유를 파악할 수 있는 것이 목표다.

---

## 1. 프로젝트 정의

**K-IFRS Financial Instruments QA** — K-IFRS 제1032호·제1039호·제1107호·제1109호를 근거로 금융상품 질문에 답하는 질의응답 시스템.

라이브: https://accounting-rag.pages.dev

### 대상 기준서

```text
금융상품
├─ 제1109호: 금융상품의 인식·분류·측정·손상·제거
├─ 제1032호: 금융상품의 부채·자본 구분과 표시
├─ 제1107호: 금융상품 공시
└─ 제1039호: 현재 적용 가능한 위험회피회계
```

네 기준서가 언급하는 외부 기준서는 참조 사실과 식별자만 보존한다. 원문이 적재되지 않은 외부 기준서 내용을 아는 것처럼 답하지 않는다.

### 제품 목표

1. 사용자의 표현이 기준서 용어와 달라도 관련 문단을 찾는다.
2. 답변을 `결론 / 판단 과정 / 근거` 형식으로 제공한다.
3. 존재하지 않는 문단을 인용하지 않는다.
4. 파싱부터 답변까지 모든 중간 결과가 원문 HWPX와 PDF 페이지로 역추적된다.

### 평가 범위

자동 벤치마크, RAG 방식별 비교, 정답률 계산은 구현하지 않는다. 사용자가 직접 질문해 평가한다.

---

## 2. 질의 파이프라인 (현재)

이것이 시스템의 핵심이다. **단계 3개, OpenAI 호출 2회**(임베딩 1 + 생성 1).

```text
질문
 │
 ▼ ① Hybrid 검색                          retrieval/hybrid.py
 │    ├ 질문을 text-embedding-3-large로 임베딩 (3,072차원)
 │    ├ Dense : chunk_embedding_vector 벡터 검색 상위 20
 │    ├ Sparse: chunk_fulltext(cjk) 전문 검색 상위 20
 │    └ weighted RRF로 결합 → 상위 12개
 │
 ▼ ② 형제 청크 보강                        retrieval/pipeline.py
 │    같은 Paragraph에서 파생된 다른 Chunk를 최대 8개 덧붙임
 │    (크기 때문에 쪼개진 문단을 다시 온전하게 만드는 목적)
 │
 ▼ ③ 답변 생성                             generation/answer.py
 │    ├ 근거 카탈로그 구성 (E1, E2, … 부여)
 │    ├ OpenAI Responses API strict JSON Schema
 │    └ 출력 즉시 결정적 검증 → 실패 시 fallback
 │
 ▼ 결론 / 판단 과정 / 근거
```

호출 주체는 `qa_pipeline.py`의 `AccountingQAPipeline.ask()` 하나다. 분기도 루프도 없다.

### 단계별 상세

**① Hybrid 검색** — `retrieval/hybrid.py`

Dense와 Sparse의 원점수는 스케일이 달라 직접 비교하지 않고 순위만 결합한다.

```python
item["rrf_score"] += weight / (rrf_k + rank)     # rrf_k = 60
```

두 Cypher 모두 아래 필터를 공유한다. 이것이 "검색 대상"의 정의다.

```cypher
WHERE coalesce(node.searchable, false) = true
  AND coalesce(node.inactive,   false) = false
  AND ($standard_id IS NULL OR node.standard_id = $standard_id)
  AND ($zone        IS NULL OR node.zone        = $zone)
```

Sparse 질의는 `escape_lucene_query()`로 Lucene 연산자(`+ - ! ( ) { } [ ] ^ " ~ * ? : \ / && ||`)만 이스케이프하고 한국어와 공백은 보존한다. `standard_id`는 `normalize_standard_id()`가 `1109` · `KIFRS1109` · `제1109호`를 모두 `"1109"`로 정규화한다.

**② 형제 청크 보강** — `retrieval/pipeline.py`

```cypher
MATCH (hit:Chunk)-[:DERIVED_FROM]->(p:Paragraph)<-[:DERIVED_FROM]-(sibling:Chunk)
WHERE hit.chunk_id IN $chunk_ids
  AND sibling.searchable = true
  AND NOT sibling.chunk_id IN $chunk_ids
  AND ($standard_id IS NULL OR sibling.standard_id = $standard_id)
  AND ($zone        IS NULL OR sibling.zone        = $zone)
LIMIT $limit
```

각 결과에 `candidate_source`가 `"hybrid"` 또는 `"sibling"`으로 붙어 어디서 왔는지 추적된다. 이 단계가 왜 필요한지는 §5에서 설명한다.

**③ 답변 생성** — `generation/answer.py`

`prepare_evidence_catalog()`가 후보를 근거 카탈로그로 바꾼다.

- `contextualized_text`를 우선 사용하고 없으면 `text`
- 개당 `max_candidate_chars`(1,800자)까지 자름
- 전체 합이 `max_context_chars`(14,000자)에 도달하면 이후 후보는 버림 ← **실질적 상한**
- `evidence_id`는 `E1`부터 순번, `citation`은 `citation_label` → `K-IFRS 제{n}호 {문단}` → `chunk_id` 순으로 결정
- 빈 텍스트 후보는 건너뛰고, 중복 `chunk_id`는 `ValueError`

`max_candidates`(20)는 `retrieval.yaml`의 `top_k`(12) + `max_siblings`(8)와 맞춰야 한다. **이 값이 더 작으면 뒤에 붙는 형제 청크가 잘려 ②단계가 무의미해진다.**

---

## 3. 답변 검증 — 게이트를 없앤 뒤 남은 유일한 방어선

게이트 4개를 제거했지만 검증이 사라진 것은 아니다. `_validate_answer()`가 모델 출력을 받자마자 결정적으로 검사한다.

| # | 검사 | 실패 예 |
|---|---|---|
| 1 | 최상위 키가 정확히 `conclusion`·`reasoning`·`evidence` 셋 | 필드 추가/누락 |
| 2 | 모든 `evidence_id`가 제공한 카탈로그에 존재 | 없는 E9를 지어냄 |
| 3 | `citation` 문자열이 제공값과 **정확히** 일치 | 출처명을 임의로 바꿈 |
| 4 | `evidence_id` 중복 없음 | 같은 근거 두 번 |
| 5 | 결론·판단과정의 모든 문장에 `[E1]` 인용이 있음 | 근거 없는 단정 |
| 6 | 인용한 ID 집합 == 반환한 근거 ID 집합 | 안 쓴 근거를 끼워 넣음 |

하나라도 어긋나면 예외가 발생하고 `_fallback()`이 실행된다.

```python
except Exception as error:
    logger.warning("answer generation fallback: %s: %s", type(error).__name__, error)
    return _fallback()   # conclusion = "근거 부족: 검증된 답변을 생성하지 못했습니다."
```

**검증되지 않은 문장은 사용자에게 도달하지 않는다.**

규칙 5의 예외: 결론이 `근거 부족:` 또는 `답변 불가:`로 시작하면 인용 없이 통과한다. 모델이 스스로 거절하는 경로를 열어두기 위한 것이다.

### 상태 판정

`qa_pipeline.py`가 이 접두사 하나로 최종 상태를 정한다.

```python
_INSUFFICIENT_PREFIXES = ("근거 부족:", "답변 불가:")
declined = conclusion.startswith(_INSUFFICIENT_PREFIXES)
```

| status | reason | 발생 조건 |
|---|---|---|
| `answered` | `sufficient` | 검증 통과 + 거절 접두사 없음 |
| `insufficient` | `self_declined` | 모델이 스스로 거절했거나 검증 실패로 fallback |
| `insufficient` | `no_evidence_found` | 검색이 0건 → 생성 단계에 도달하지 않음 |

프론트엔드는 이 조합으로만 화면을 고른다. **conclusion 문자열을 파싱해 상태를 추측하지 않는다.**

---

## 4. 왜 이 구조가 되었나 — 제거한 것과 그 이유

2026-07-29 이전에는 게이트 4개와 LLM 호출 최대 12회짜리 파이프라인이었다.

```text
질문 → [LLM 질문분석] → Hybrid → 그래프확장 → 후보병합
     → [LLM 재정렬 ×4배치] → ⛔기계판정(6종) → ⛔[LLM 의미판정]
     → (부족하면 질문 보강 후 위 전체 재실행, 최대 1회)
     → ⛔근거 필터 → [LLM 답변생성] → ⛔인용검증 → 답변
```

### 무엇이 문제였나

라이브 백엔드에 질문 6개를 던져 측정한 결과 **4개가 기각**됐다. 기각된 것 중에는 "위험회피회계를 적용하기 위한 요건은?" 같은 핵심 질문도 있었다.

**원인 A — 의미 판정이 완전성을 요구했다**

```text
Return sufficient=true only when all material claims, conditions,
and exceptions needed for a grounded answer are covered.
                    ↑ 예외까지 전부 커버돼야 통과
```

모델은 시킨 대로 했다. 실제 기각 사유:

> "문단 16A~16D의 예외적 지분분류의 **모든 특성 및 구체적 조건**이 제공되지 않았다"
> "문단 25의 '다음의 경우를 제외'하는 **예외 내용**이 제공되지 않았다"

K-IFRS는 어떤 주제든 예외 규정이 딸려 있다. 상위 10개 청크에 그게 전부 들어올 일이 거의 없으므로 **정상 질문 대부분이 기각되는 구조**였다.

**원인 B — 같은 질문이 실행마다 다른 결과를 냈다**

```text
"위험회피회계를 적용하기 위한 요건은?"
  1차 → insufficient / deterministic_failed / 근거 0
  2차 → answered / sufficient / 완전한 4단락 답변
```

기계 판정은 재정렬 LLM이 매긴 점수(`question_relevance ≥ 3` AND `direct_evidence ≥ 3`)에 의존했는데, 5개 LLM 호출 어디에도 `temperature`·`seed`·`top_p` 설정이 없었다. 점수가 흔들리니 경계선 질문이 통과와 기각 사이를 오갔다.

### 어떻게 고쳤나

임계값을 재조정하는 대신 **판정 단계 자체를 제거**했다. 답변 생성 프롬프트에 이미 자체 거절 규칙이 있었기 때문이다.

```text
5. 결론에 필요한 조건, 예외 또는 사실관계가 부족하면 단정하지 않는다.
   그 문장은 '근거 부족:'으로 시작하여 무엇이 부족한지 구체적으로 밝힌다.
6. 질문에 답할 근거가 전혀 없으면 결론에서 답변할 수 없다고 밝히고
   evidence는 빈 배열로 둔다.
```

판정 LLM, 재검색 루프, 기계 검사 6종이 이 신호 하나로 대체됐다.

### 결과

| | 이전 | 현재 |
|---|---|---|
| LLM 호출 | 최대 12회 | **1회** (+ 임베딩 1) |
| 거절 게이트 | 4개 | **0개** |
| 검색 단계 | 6단계 | **2단계** |
| 튜닝 파라미터 | 20개+ | **7개** |
| 코드 | — | **약 1,400줄 제거** |
| 답변 성공 (동일 질문 6개) | **2/6** | **5/6** |

제거한 모듈: `retrieval/reranker.py`(289줄) · `generation/sufficiency.py`(357줄) · `generation/citation_verifier.py`(181줄) · `retrieval/graph_expansion.py`(355줄) · `query/analysis.py`(193줄) · `config/query.yaml`

측정 결과 (Aura + OpenAI 실측):

| 질문 | 이전 | 현재 |
|---|---|---|
| 금융자산 상각후원가 측정 조건 | answered | answered |
| 위험회피회계 적용 요건 | deterministic_failed | **answered** |
| 기대신용손실 (키워드 한 단어) | semantic_insufficient | **answered** (근거 3) |
| 금융부채랑 자본 어떻게 구분해? | semantic_insufficient | **answered** (근거 2) |
| 금융자산의 정의 | — | answered |
| 오늘 서울 날씨 어때? | 거절 | **거절** (정상) |

마지막 항목이 중요하다. **범위 밖 질문은 여전히 걸러진다.** 기각률이 낮아진 것이지 아무거나 답하게 된 것이 아니다.

"금융부채랑 자본 어떻게 구분해?"는 답을 주면서 한계도 함께 밝혔다 — *"다만, 제공된 근거에는 지분상품의 일반적 정의와 자기지분상품 결제계약에 관한 전체 판단요건이…"*. 전면 거절 대신 뒷받침되는 만큼 답하고 빈틈을 알리는 것이 의도한 동작이다.

### 무엇을 잃었나

- 답변 문장이 근거 원문의 **내용**과 일치하는지 판단하는 장치가 없다. 형식(ID·citation 일치)만 검사한다.
- 답변이 근거의 범위를 넘어 일반화하는지 감지하지 못한다. 이전의 의미 판정이 그 역할을 일부 했다.

이 트레이드오프는 의도적이다. 기각률 60%는 서비스가 성립하지 않는 수준이었다.

---

## 5. 하위항목(⑴ → ㈎ → ①)을 다루는 방식

기준서 문단은 계층적 하위항목을 갖는다. 검색 시점에 그래프로 부모를 복원할 수도 있었지만, **청킹 단계에서 이미 해결되어 있어 불필요**하다.

### 실제 저장 형태

```text
청크 KIFRS1032-11-C01 (834자, 원문 노드 17개를 덮음)

  금융자산은 다음의 자산을 말한다.          ← 부모 줄기가 함께 들어 있다
  ⑴ 현금
  ⑵ 다른 기업의 지분상품
  ⑶ 다음 중 어느 하나에 해당하는 계약상 권리
    ㈎ 거래상대방에게서 현금 등 금융자산을 수취할 계약상 권리
    ㈏ 잠재적으로 유리한 조건으로 …
  ⑷ 기업 자신의 지분상품으로 결제하거나 …
    ㈎ 수취할 자기지분상품의 수량이 변동 가능한 비파생상품
    ㈏ 확정 수량의 자기지분상품을 … 파생상품
      ① 문단 16A와 16B에 따라 지분상품으로 분류하는 풋가능 금융상품
      ② 발행자가 청산하는 경우에만 …
      ③ 자기지분상품을 미래에 수취하거나 인도하기 위한 계약인 금융상품
```

### 근거 (Aura 실측)

| 확인 항목 | 값 | 의미 |
|---|---|---|
| `Subparagraph -CONTAINS-> Subparagraph` | **0건** | 파서가 평탄화했다. 중첩 깊이는 1뿐 |
| 청크당 `DERIVED_FROM` 엣지 | **최소 2개** (최대 37) | 조각만 담은 청크가 없다 |
| 한 문단이 청크 1개에 온전히 들어간 비율 | **3,701 / 3,831 = 96.6%** | 대부분 이미 완전하다 |

### 남은 3.4%가 형제 보강의 이유

문단이 길면 크기 제한 때문에 여러 청크로 갈린다. `KIFRS1032-11`도 C01(834자, 금융자산 정의) + C02(557자, 금융부채 정의 뒷부분)로 나뉜다. "금융자산의 정의"를 물으면 C01만 걸리고 C02는 놓친다.

| 분할 청크 수 | 문단 수 |
|---|---:|
| 1 | 3,701 |
| 2 | 81 |
| 3 | 28 |
| 4 | 12 |
| 5 | 6 |
| 6 | 2 |
| 7 | 1 |

형제까지 합쳤을 때 글자수: 평균 1,283자 · p95 3,326자 · 최대 6,250자. `max_context_chars`(14,000) 안에 들어온다.

---

## 6. 설계 결정 요약

| 항목 | 확정 내용 | 근거 |
|---|---|---|
| 데이터베이스 | Neo4j 하나 | 구조 그래프·벡터 인덱스·전문 검색을 한 저장소에서 관리 |
| 생성 모델 | `.env`의 `OPENAI_CHAT_MODEL` | 코드 하드코딩 없이 교체 |
| 임베딩 모델 | `.env`의 `OPENAI_EMBEDDING_MODEL` | 차원과 인덱스를 설정으로 고정 |
| 재정렬 모델 | **제거** | 점수 변동성이 게이트를 흔들었고, 게이트 제거로 존재 이유가 사라짐 |
| Hybrid 결합 | Dense + Sparse를 weighted RRF | 스케일이 다른 점수를 직접 비교하지 않고 순위만 결합 |
| 그래프 확장 | **제거**, `DERIVED_FROM` 형제 조회만 유지 | 확장 결과 40개 중 대부분이 재정렬에서 탈락했고, 실제로 필요한 것은 쪼개진 문단 복원뿐이었음 |
| 충분성 검사 | **제거**, 답변 생성기가 자체 판단 | 완전성 기준이 정상 질문을 기각. 프롬프트에 이미 거절 규칙 존재 |
| 인용 검증 | 별도 모듈 제거, 생성기 내부 검증 유지 | 검증 로직은 생성 직후가 가장 자연스러운 위치 |
| 답변 출력 | 결론 / 판단 과정 / 근거 | 결론과 논리, 원문 근거를 각각 검증 가능 |
| 자동 평가 | 구현하지 않음 | 사용자가 직접 평가 |

### 프레임워크를 쓰지 않은 이유

LangChain·LangGraph를 사용하지 않고 Neo4j Python Driver와 OpenAI SDK를 직접 호출한다. 의존성에도 없다.

파이프라인이 분기 없는 3단계 직선이고 상태 전이가 없다. 프레임워크가 주는 추상화보다 호출 지점을 직접 보는 편이 디버깅에 유리하다. 실제로 "왜 기각되는가"를 추적할 때 각 LLM 호출의 입출력을 그대로 볼 수 있어 §4의 원인 A·B를 빠르게 찾았다.

LLM이 임의의 Cypher를 생성하게 하지 않는다. 검색은 관계 유형·방향·깊이가 고정된 Cypher 템플릿만 쓴다.

---

## 7. 기술 스택

| 영역 | 기술 | 역할 |
|---|---|---|
| 언어 | Python 3.11+ | 파싱, 적재, 검색, API |
| LLM | OpenAI Responses API | 답변 생성 (strict JSON Schema), 이미지 판독 |
| Embedding | `text-embedding-3-large` 3,072차원 | 질문·Chunk 벡터 |
| 데이터베이스 | Neo4j 5.x (로컬 / AuraDB Free) | 그래프·벡터·전문 검색 통합 |
| DB 접근 | Neo4j Python Driver, 명시적 Cypher | 재현 가능한 적재와 제한된 탐색 |
| RAG | 프레임워크 없이 직접 구현 | Hybrid 결합, 형제 보강, 답변 검증 |
| API | FastAPI | 동기·비동기 질의 |
| 검증 | Pydantic | JSONL, LLM Structured Output, API 스키마 |
| 테스트 | pytest | 파서·청킹·그래프·검색·답변 회귀 |

---

## 8. 디렉터리 구조 (실제)

```text
accounting_rag/
├── README.md                    프로젝트 개요와 질의 흐름
├── PROJECT_STRUCTURE.md         이 문서
├── NEO4J_SCHEMA.md              노드·관계·ID·인덱스
├── DEPLOYMENT.md                Aura·Render·Pages 배포 절차
├── REVIEW_DECISIONS.md          코드 리뷰 결정 기록
├── PROJECT_WORKFLOW.html        색인·질의 흐름 시각화
├── Dockerfile                   Render 배포용
├── pyproject.toml
├── .env / .env.example
│
├── config/
│   ├── retrieval.yaml           hybrid(top-k, RRF) + retrieval(top_k, max_siblings)
│   ├── answering.yaml           answer(근거 개수·글자수·토큰 예산)
│   ├── graph_schema.yaml        노드·관계 정의 (적재·검증용)
│   ├── chunking.json            청킹 임계값
│   ├── embedding.json           임베딩 모델·차원
│   └── page_mapping.json        PDF 페이지 매핑 설정
│
├── db/
│   ├── schema.cypher            제약조건·인덱스 DDL
│   └── migrations/002_rebuild_chunk_fulltext_cjk.cypher
│
├── data/                        (Git 제외)
│   ├── raw/standards/           HWPX·PDF 원본
│   ├── processed/               paragraphs·blocks·tables·footnotes·references JSONL
│   ├── chunks/                  chunks.jsonl + 품질 보고서
│   ├── embeddings/              임베딩 캐시
│   └── semantic/                Concept·MENTIONS JSONL
│
├── src/accounting_rag/
│   ├── qa_pipeline.py           ★ 검색 → 생성 → 상태 판정
│   ├── api/
│   │   ├── app.py               FastAPI 라우트, job 저장소, rate limit
│   │   ├── dependencies.py      파이프라인 조립, 이미지 판독
│   │   └── schemas.py           요청·응답 Pydantic
│   │   └── static/              index.html · app.js · styles.css · favicon.png
│   ├── retrieval/
│   │   ├── hybrid.py            ★ Dense + Sparse + RRF
│   │   ├── pipeline.py          ★ 형제 청크 보강
│   │   └── embeddings.py        임베딩 캐시 생성
│   ├── generation/
│   │   └── answer.py            ★ 답변 생성 + 결정적 검증
│   ├── ingestion/
│   │   ├── hwpx_parser.py       HWPX XML 파싱
│   │   ├── dataset_builder.py   JSONL 생성
│   │   ├── chunk_builder.py     검색용 Chunk 생성
│   │   └── pdf_page_mapper.py   PDF 페이지 매핑
│   └── graph/
│       ├── loader.py            Neo4j 적재
│       ├── semantic_builder.py  Concept·MENTIONS 생성
│       └── semantic_loader.py   의미 KG 적재
│
├── scripts/                     각 단계 실행 + validate_* 짝
└── tests/                       pytest
```

★ 표시가 질의 시점에 실행되는 코드다. 나머지는 색인 파이프라인이거나 API 계층이다.

---

## 9. 원문 구조와 청킹 정책

### 원문 계층

```text
Standard
└─ Zone
   └─ Section
      └─ Paragraph
         ├─ Subparagraph      (깊이 1, 중첩 없음)
         ├─ Table
         └─ Footnote
```

공식 문단번호가 가장 강한 경계다. 문단 내부의 `⑴`, `㈎`, `①`, `가.`는 소분류로 인식하며 **원문 계층과 검색 단위(Chunk)는 분리**한다.

### 모든 내용 반영 정책

| 영역 | 처리 |
|---|---|
| 공식 본문 | 기본 검색의 최우선 근거 |
| 부록·적용지침·용어 정의 | 본문 다음 우선순위 |
| 적용사례·실무적용지침 | 계산과 사실관계의 보조 근거 |
| 결론도출근거·소수의견 | 해석 배경이 필요할 때만 |
| 표지·저작권·목차·빈 문단 | 파싱·보존하되 `searchable=false` |
| 삭제 표시·시행일·경과규정 | 버전 판단용 별도 속성 |

### Chunk 생성 원칙

1. 원본 `Paragraph`·`Subparagraph`는 수정하지 않는다.
2. 기본 Chunk는 공식 문단 하나다.
3. 문단이 길면 하위 소분류 경계에서 분할한다.
4. 짧은 문단도 다른 문단과 자동 병합하지 않는다. (인용 경계 유지)
5. 표는 구조화 데이터와 검색용 직렬화 텍스트를 모두 만든다.
6. 각주는 본문 Chunk에 문맥으로 포함하고 별도 원문 노드도 유지한다.
7. 고정 overlap 대신 `NEXT`와 계층 관계로 문맥을 확장한다.

3,904개 문단 실측 분포를 근거로 임계값을 정했다.

| 지표 | 값 |
|---|---:|
| 문단 원문 길이 중앙값 | 250자 |
| 95백분위 | 656자 |
| 99백분위 | 1,224자 |
| 1,200자 초과 문단 | 40개 |
| 소분류 | 2,125개 |
| continuation이 있는 문단 | 447개 |
| 가장 긴 문단 | 4,554자 |
| 가장 긴 표 | 6,653자 |

현재 설정: 문단 목표/최대 **900/1,200자**, 표 목표/최대 **1,200/1,800자**. 긴 문단은 Block·Subparagraph 경계에서, 표는 행 경계에서 분할한다.

### Chunk 메타데이터

```json
{
  "chunk_id": "KIFRS1109-5.5.17-C01",
  "text": "검색에 사용할 텍스트",
  "contextualized_text": "절 제목 등 문맥을 덧붙인 텍스트",
  "standard_id": "1109",
  "zone": "standard_body",
  "section_path": ["손상", "기대신용손실"],
  "source_paragraph_ids": ["KIFRS1109-5.5.17"],
  "source_subparagraph_ids": [],
  "table_ids": [], "footnote_ids": [],
  "pdf_page_start": 120, "pdf_page_end": 121,
  "chunk_order": 1,
  "searchable": true, "inactive": false,
  "search_priority": 100,
  "citation_label": "K-IFRS 제1109호 문단 5.5.17"
}
```

`Chunk`는 검색용 파생 데이터이고 법적 원문 단위가 아니다. 따라서 반드시 `DERIVED_FROM`으로 원문 노드에 연결한다. **이 관계가 형제 보강의 기반**이다.

---

## 10. 데이터 현황 (Neo4j AuraDB Free 실측)

노드 22,018개 · 관계 92,121개 · 검색 대상 Chunk 4,144개

| 노드 | 개수 | | 관계 | 개수 | 질의에 사용 |
|---|---:|---|---|---:|:---:|
| Block | 8,533 | | APPEARS_ON | 19,968 | |
| Chunk | 4,379 | | MENTIONS | 17,712 | |
| Paragraph | 3,904 | | **DERIVED_FROM** | 14,087 | **○** |
| Subparagraph | 2,125 | | REFERS_TO | 12,796 | |
| PdfPage | 1,671 | | NEXT | 11,435 | |
| Section | 713 | | CONTAINS | 8,433 | |
| Table | 362 | | HAS_BLOCK | 6,674 | |
| Footnote | 227 | | HAS_TABLE | 574 | |
| Concept | 45 | | HAS_FOOTNOTE | 442 | |
| ExternalStandard | 35 | | | | |
| Zone | 20 | | | | |
| Standard | 4 | | | | |

인덱스 (모두 ONLINE):

| 이름 | 유형 | 질의에 사용 |
|---|---|:---:|
| `chunk_embedding_vector` | VECTOR, 3,072차원 cosine | **○** |
| `chunk_fulltext` | FULLTEXT, `cjk` analyzer | **○** |
| `concept_fulltext` | FULLTEXT | |

**질의 시점에 읽는 것은 인덱스 2개와 관계 1종뿐이다.** 나머지는 색인 파이프라인의 산출물이며 적재·검증되어 있지만 현재 검색 경로에서 사용하지 않는다. 향후 확장 여지로 남긴다.

---

## 11. 색인 파이프라인 (사전 1회)

```text
HWPX 원본 4개 ─────────────────────────── PDF 원문
      │                                      │
      ▼ parse_all_standards.py               ▼ map_pdf_pages.py
      │  문단·블록·표·각주·참조 추출          │  1,671쪽 매핑
      │  → data/processed/*.jsonl            │
      │                                      │
      ▼ build_chunks.py ◄────────────────────┘
      │  검색용 Chunk 4,379개 → data/chunks/
      │
      ├──────────────────────┬───────────────────────┐
      ▼ build_embeddings.py  ▼ build_semantic_kg.py  ▼ load_neo4j.py
        3,072차원 임베딩       Concept·MENTIONS         구조·참조 그래프
        → data/embeddings/    → data/semantic/
      │                      │                       │
      ▼ load_embeddings_neo4j ▼ load_semantic_neo4j  │
      └──────────────────────┴───────────────────────┘
                             ▼
                          Neo4j
```

각 단계마다 `validate_*.py`가 짝으로 있고 품질 보고서를 `data/**/*_QUALITY_REPORT.md`에 남긴다.

| 스크립트 | 검증 내용 |
|---|---|
| `validate_processed_dataset.py` | JSONL 무결성, 참조 해석 상태 |
| `validate_chunks.py` | ID 중복, 길이 위반, 원문 역추적 |
| `validate_page_mapping.py` | PDF 페이지 매핑 신뢰도 |
| `validate_embeddings.py` | 차원, 개수, 캐시 일치 |
| `validate_neo4j.py` | 노드·관계 개수, 인덱스 ONLINE, analyzer |
| `validate_semantic_kg.py` | Concept 개수, 정의 출처, ECL 경로 |

---

## 12. API

| 엔드포인트 | 용도 |
|---|---|
| `GET /` | 웹 UI (정적) |
| `GET /health` | 상태 확인 |
| `GET /favicon.ico` | 파비콘 (백엔드 자체 서빙용) |
| `POST /v1/ask` | 동기 질의 |
| `POST /v1/jobs` | 비동기 접수 → `202` + `job_id` |
| `GET /v1/jobs/{id}` | 폴링 (`pending` / `complete` / `error`) |
| `DELETE /v1/jobs/{id}` | 조기 폐기 |

### 비동기 job

답변에 30~60초가 걸리고 Render 콜드 스타트가 최대 1분이라, 브라우저 연결이 끊겨도 결과를 잃지 않도록 작업을 분리했다.

- job은 프로세스 메모리에 **10분** 보관, 만료분 자동 정리
- `request_id`(UUID)를 보내면 **멱등** 처리 — 같은 id는 같은 job을 반환하고 worker를 다시 만들지 않는다
- rate limit은 job이 실제로 생성될 때만 소비된다 (`jobs.create(request_id, lambda: check_rate_limit(...))`)

제약: JobStore는 프로세스 메모리 기반이라 여러 서버 프로세스 간에 공유되지 않고 재시작을 넘어 보장하지 않는다. 필요해지면 Redis 등으로 옮겨야 한다.

### 이미지 첨부

첨부 이미지는 OpenAI 비전 모델로 문구·숫자·표 구조를 텍스트로 옮긴 뒤 질문에 덧붙여 **기존 검색 경로를 그대로 탄다**. 서버에 저장하지 않고 OpenAI에도 `store=False`로 전달한다. PNG·JPEG·WEBP·GIF, 최대 4장, 장당 5MB. 서버가 base64 유효성과 크기를 다시 검증한다.

---

## 13. 환경변수와 설정

```env
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL=gpt-5.6-terra
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

`OPENAI_RERANK_MODEL`은 재정렬 제거로 더 이상 사용하지 않는다.

배포 전용 선택 항목 (미설정 시 로컬 동작 불변):

```env
CORS_ALLOW_ORIGINS=https://your-site.pages.dev   # 쉼표 구분. 미설정 시 CORS 헤더 없음
ASK_RATE_LIMIT_PER_HOUR=10                       # IP당 시간당. 0이면 무제한
TRUST_PROXY_HEADERS=1                            # 리버스 프록시 뒤에서만
```

### `config/retrieval.yaml`

```yaml
hybrid:
  dense_top_k: 20        # Dense 벡터 검색 후보
  sparse_top_k: 20       # Sparse 전문 검색 후보
  seed_top_k: 12         # RRF 결합 후 상위
  rrf_k: 60              # RRF 상수
  dense_weight: 1.0
  sparse_weight: 1.0
  embedding_dimensions: 3072

retrieval:
  top_k: 12              # 답변 생성기로 넘길 seed 수
  max_siblings: 8        # 형제 청크 보강 상한
```

### `config/answering.yaml`

```yaml
answer:
  max_candidates: 20        # ★ retrieval의 top_k + max_siblings 이상이어야 함
  max_candidate_chars: 1800 # 근거 하나에서 잘라 쓸 최대
  max_context_chars: 14000  # ★ 실질적 상한. 이 값이 먼저 걸린다
  max_question_chars: 2000  # 초과 시 ValueError (이미지 추출 텍스트 포함)
  max_output_tokens: 3000
```

**`max_candidates`가 `top_k + max_siblings`보다 작으면 형제 청크가 잘려 나가 §5의 보강이 무의미해진다.** 두 파일을 함께 봐야 한다.

---

## 14. 구현 이력

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 원문 확보와 1109호 파서 시제품, 수동 검수 | 완료 |
| 2 | 네 기준서 전체 구조화·청킹 (문단 3,904 · 블록 8,533 · 표 362 · 각주 227 · 참조 7,450 · Chunk 4,379) | 완료 |
| 3 | Neo4j 스키마와 구조·참조 적재, `cjk` analyzer 재구축, Concept 45개 | 완료 |
| 4 | 임베딩 4,144개 생성·적재, Hybrid 검색과 weighted RRF | 완료 |
| 5 | GraphRAG와 OpenAI 재정렬 | 완료 후 **제거** (§4) |
| 6 | 질문 분석·충분성 검사·인용 검증을 포함한 답변 파이프라인 | 완료 후 **일부 제거** (§4) |
| 7 | FastAPI와 웹 UI | 완료 |
| 8 | 문서화, 공개 저장소 정리 | 완료 |
| 9 | 무료 클라우드 배포 (Aura + Render + Pages) | 완료 |
| 10 | 비동기 job API와 이미지 첨부 | 완료 |
| 11 | UI 재설계 (대화 기록·복사·스크롤 보존) | 완료 |
| 12 | **RAG 파이프라인 단순화** (§4) | 완료 |

### 시도했다가 제거한 것

**의미 규칙 KG** — `Rule`·`Condition`·`Exception`·`Example` 후보 추출기와 사람 검토 UI를 구현했으나 제거했다. 후보를 실제로 생성·승인한 적이 없어 검색에 쓰인 적이 없고, 승인 절차를 운영할 계획도 없었다. `config/graph_schema.yaml`의 노드·관계 정의도 함께 삭제했다.

**GraphRAG 확장** — Hybrid Seed에서 허용 관계 7종을 1~2 hop 확장해 근거 40개를 만들었다. 그중 대부분이 재정렬에서 탈락했고, 실제로 필요했던 것은 "쪼개진 문단 복원" 하나였다. 그 하나를 `DERIVED_FROM` 형제 조회 4줄로 대체했다.

**OpenAI 재정렬** — 후보에 4개 축(질문 관련성 40% · 본문 우선도 20% · 직접 근거성 30% · 참조 연결성 10%) 점수를 매겨 상위 10개를 골랐다. `temperature` 미설정으로 점수가 흔들려 하류의 기계 판정을 불안정하게 만든 원인이었다.

**충분성 검사 2단계** — 기계 검사 6종 + 의미 판정 LLM. §4의 원인 A.

---

## 15. 알려진 한계

| 한계 | 내용 |
|---|---|
| **인용 검증이 형식만 본다** | `evidence_id` 존재와 `citation` 일치는 확인하지만, 답변 문장이 근거 원문의 내용과 실제로 일치하는지는 검증하지 않는다 |
| **표현 변경을 허용한다** | 원문 복사를 요구하지 않으므로 의미를 왜곡한 요약이 걸러지지 않는다 |
| **범위 초과를 감지하지 못한다** | 답변이 근거 밖으로 일반화해도 막을 장치가 없다 |
| **결과에 변동성이 있다** | 답변 생성 LLM에 `temperature`를 설정하지 않았다. 재정렬 제거로 변동 폭은 크게 줄었지만 0은 아니다 |
| **그래프 대부분이 유휴** | `REFERS_TO` 12,796개, `MENTIONS` 17,712개는 적재만 되어 있다 |
| **정의 질문의 검색 편향** | 부록 정의표는 여러 용어를 한 Chunk에 담아, 해당 용어를 자주 언급하는 본문에 순위가 밀릴 수 있다. 정의 영역 가산점이나 용어 단위 재청킹으로 개선 가능 |
| **콜드 스타트** | Render 무료 플랜은 15분 유휴 후 슬립. 첫 요청 최대 1분 |
| **job 저장소가 프로세스 로컬** | 다중 프로세스·재시작을 넘어 멱등성을 보장하지 않는다 |

---

## 16. 다음에 이 코드를 만질 때

1. **`reason` 값을 바꾸면 프론트엔드도 같이 바꾼다.** `api/schemas.py`의 `AnswerReason`과 `static/app.js`의 `resultKind()`가 짝이다. 프론트엔드를 먼저 배포해 구·신 값을 모두 인식하게 한 뒤 백엔드를 올린다. 과거에 백엔드만 뒤처져 모든 답변이 오류로 표시된 적이 있다.
2. **`conclusion` 문자열로 상태를 판별하지 않는다.** `status`·`reason` 조합만 쓴다.
3. **`config/retrieval.yaml`과 `config/answering.yaml`은 함께 본다.** `max_candidates ≥ top_k + max_siblings`.
4. **`static/`에 자산을 추가하면 `scripts/build_frontend_dist.py`의 복사 목록에도 추가한다.** 누락하면 Pages 번들에 들어가지 않는다.
5. **로컬 검증만으로 프론트엔드 결론을 내지 않는다.** 로컬은 FastAPI가 정적 파일과 라우트를 함께 제공하지만 실제 프론트엔드는 라우트가 없는 정적 호스팅이다.
6. **게이트를 다시 추가하려면 기각률을 먼저 측정한다.** §4의 6개 질문 세트가 기준선이다.
