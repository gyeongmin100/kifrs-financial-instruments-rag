# K-IFRS Financial Instruments RAG

K-IFRS 금융상품 기준서 4개(제1032호·제1039호·제1107호·제1109호)를 구조화해 **Neo4j 하나**에 적재하고, 기준서 원문에 근거해서만 답하는 질의응답 시스템입니다.

라이브 서비스: https://accounting-rag.pages.dev

이 저장소는 **RAG 구현부**만 담고 있습니다 (웹 UI·API 계층 제외).

LangChain·LangGraph 없이 Neo4j Driver와 OpenAI SDK를 직접 호출합니다. 이유는 [아래](#프레임워크를-쓰지-않은-이유)에 정리했습니다.

## 질의 흐름

단계 3개, OpenAI 호출 2회(임베딩 1 + 생성 1), 거절 게이트 0개입니다.

```text
질문
 │
 ▼ ① Hybrid 검색                         retrieval/hybrid.py
 │     Dense 벡터 top 20 ─┐
 │                        ├─ weighted RRF ─→ 상위 12개
 │     CJK Sparse  top 20 ─┘
 │
 ▼ ② 형제 청크 보강                       retrieval/pipeline.py
 │     같은 Paragraph에서 나온 나머지 Chunk를 최대 8개까지 덧붙임
 │
 ▼ ③ 답변 생성                            generation/answer.py
 │     근거로 답할 수 없으면 모델이 결론을 '근거 부족:'으로 시작
 │
 ▼ 결론 / 판단 과정 / 근거
```

## 이 프로젝트에서 실제로 배운 것

처음에는 게이트 4개(기계 판정 6종 · 의미 판정 LLM · 근거 필터 · 인용 검증)와 LLM 호출 최대 12회짜리 파이프라인이었습니다. **정상 질문의 60%가 기각됐습니다.**

라이브 백엔드에 질문 6개를 던져 원인을 찾았습니다.

**원인 A — 판정 프롬프트가 완전성을 요구했다**

```text
Return sufficient=true only when all material claims, conditions,
and exceptions needed for a grounded answer are covered.
```

모델은 시킨 대로 했습니다. 실제 기각 사유:

> "문단 16A~16D의 예외적 지분분류의 **모든 특성 및 구체적 조건**이 제공되지 않았다"

K-IFRS는 어떤 주제든 예외 규정이 딸려 있습니다. 상위 10개 청크에 그게 전부 들어올 일이 없으니 구조적으로 통과가 불가능했습니다.

**원인 B — 같은 질문이 실행마다 다른 결과를 냈다**

```text
"위험회피회계를 적용하기 위한 요건은?"
  1차 → insufficient / deterministic_failed / 근거 0
  2차 → answered / sufficient / 완전한 4단락 답변
```

기계 판정이 재정렬 LLM의 점수에 의존했는데, LLM 호출 5곳 어디에도 `temperature`·`seed`·`top_p` 설정이 없었습니다.

**조치 — 임계값 재조정 대신 판정 단계 제거**

답변 생성 프롬프트에 이미 자체 거절 규칙이 있었습니다.

```text
5. 결론에 필요한 조건, 예외 또는 사실관계가 부족하면 단정하지 않는다.
   그 문장은 '근거 부족:'으로 시작하여 무엇이 부족한지 구체적으로 밝힌다.
```

| | 이전 | 현재 |
|---|---|---|
| LLM 호출 | 최대 12회 | **1회** |
| 거절 게이트 | 4개 | **0개** |
| 검색 단계 | 6단계 | **2단계** |
| 코드 | — | **약 1,400줄 제거** |
| 답변 성공 (동일 질문 6개) | **2/6** | **5/6** |

| 질문 | 이전 | 현재 |
|---|---|---|
| 금융자산 상각후원가 측정 조건 | answered | answered |
| 위험회피회계 적용 요건 | deterministic_failed | **answered** |
| 기대신용손실 (키워드 한 단어) | semantic_insufficient | **answered** |
| 금융부채랑 자본 어떻게 구분해? | semantic_insufficient | **answered** |
| 금융자산의 정의 | — | answered |
| 오늘 서울 날씨 어때? | 거절 | **거절** (정상) |

마지막 행이 중요합니다. **범위 밖 질문은 여전히 걸러집니다.** 기각률이 낮아진 것이지 아무거나 답하게 된 것이 아닙니다.

제거한 모듈: `reranker.py`(289줄) · `sufficiency.py`(357줄) · `citation_verifier.py`(181줄) · `graph_expansion.py`(355줄) · `query/analysis.py`(193줄)

## 헛소리를 막는 장치

게이트를 없앴다고 검증이 사라진 것은 아닙니다. `generation/answer.py`의 `_validate_answer()`가 모델 출력을 받자마자 결정적으로 검사합니다.

| # | 검사 | 실패 예 |
|---|---|---|
| 1 | 최상위 키가 정확히 `conclusion`·`reasoning`·`evidence` 셋 | 필드 추가/누락 |
| 2 | 모든 `evidence_id`가 제공한 카탈로그에 존재 | 없는 E9를 지어냄 |
| 3 | `citation` 문자열이 제공값과 **정확히** 일치 | 출처명을 임의로 바꿈 |
| 4 | `evidence_id` 중복 없음 | 같은 근거 두 번 |
| 5 | 결론·판단과정의 모든 문장에 `[E1]` 인용 | 근거 없는 단정 |
| 6 | 인용한 ID 집합 == 반환한 근거 ID 집합 | 안 쓴 근거를 끼워 넣음 |

하나라도 어긋나면 `_fallback()`이 실행되어 생성 답변을 버립니다. **검증되지 않은 문장은 사용자에게 도달하지 않습니다.**

인용 검증을 "없앤" 것이 아니라 별도 모듈에서 생성 직후로 옮긴 것입니다.

## 하위항목(⑴ → ㈎ → ①)을 다루는 방식

기준서 문단은 계층적 하위항목을 갖습니다. 검색 시점에 그래프로 부모를 복원할 수도 있었지만, **청킹 단계에서 이미 해결되어 있어 불필요**했습니다.

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

Neo4j 실측 근거:

| 확인 항목 | 값 |
|---|---|
| `Subparagraph -CONTAINS-> Subparagraph` 중첩 | **0건** (파서가 평탄화, 깊이 1뿐) |
| 청크당 `DERIVED_FROM` 엣지 | **최소 2개** (조각만 담은 청크 없음) |
| 한 문단이 청크 1개에 온전히 들어간 비율 | **3,701 / 3,831 = 96.6%** |

**남은 3.4%가 형제 청크 보강(②단계)의 이유입니다.** 문단이 길면 크기 제한 때문에 여러 청크로 갈리는데, 그때 뒷부분만 검색에 걸리면 앞부분 없이 답변이 만들어집니다.

```cypher
MATCH (hit:Chunk)-[:DERIVED_FROM]->(p:Paragraph)<-[:DERIVED_FROM]-(sibling:Chunk)
WHERE hit.chunk_id IN $chunk_ids
  AND sibling.searchable = true
  AND NOT sibling.chunk_id IN $chunk_ids
```

그래프 확장 355줄이 이 4줄로 대체됐습니다.

## Hybrid 검색

Dense와 Sparse의 원점수는 스케일이 달라 직접 비교하지 않고 순위만 결합합니다.

```python
item["rrf_score"] += weight / (rrf_k + rank)     # rrf_k = 60
```

- **Dense**: `text-embedding-3-large` 3,072차원 → Neo4j Vector Index (cosine)
- **Sparse**: Neo4j Full-text Index, **`cjk` analyzer**. `standard-no-stop-words`는 한국어 질의에서 관련 없는 결과를 반환하는 것이 실측되어 사용하지 않습니다.
- 두 채널 모두 `searchable=true AND inactive=false` 필터를 적용합니다. Neo4j 전문 인덱스 자체는 이 조건을 강제하지 않습니다.

`normalize_standard_id()`가 `1109` · `KIFRS1109` · `제1109호`를 모두 `"1109"`로 정규화하고, `escape_lucene_query()`가 Lucene 연산자만 이스케이프하며 한국어와 공백은 보존합니다.

## 색인 파이프라인

```text
HWPX 원본 4개
 ▼ parse_all_standards.py     문단·블록·표·각주·참조 추출
 ▼ map_pdf_pages.py           PDF 1,671쪽과 문단 매핑
 ▼ build_chunks.py            검색용 Chunk 생성
 ▼ build_embeddings.py        3,072차원 임베딩
 ▼ build_semantic_kg.py       공식 정의 기반 Concept·MENTIONS
 ▼ load_neo4j.py / load_semantic_neo4j.py / load_embeddings_neo4j.py
```

각 단계마다 `validate_*.py`가 짝으로 있고 품질 보고서를 남깁니다.

### 청킹 정책

3,904개 문단 실측 분포를 근거로 임계값을 정했습니다.

| 지표 | 값 |
|---|---:|
| 문단 원문 길이 중앙값 | 250자 |
| 95백분위 | 656자 |
| 99백분위 | 1,224자 |
| 소분류 | 2,125개 |
| 가장 긴 문단 | 4,554자 |

문단 목표/최대 **900/1,200자**, 표 목표/최대 **1,200/1,800자**. 긴 문단은 Block·Subparagraph 경계에서, 표는 행 경계에서 분할합니다. **짧은 문단도 자동 병합하지 않습니다** — 문단 인용 경계를 유지하기 위해서입니다.

## 데이터 현황 (Neo4j 실측)

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

**질의 시점에 읽는 것은 인덱스 2개(`chunk_embedding_vector`, `chunk_fulltext`)와 관계 1종(`DERIVED_FROM`)뿐입니다.** 나머지 그래프는 적재·검증되어 있지만 현재 검색 경로에서 사용하지 않으며, 원문 추적과 향후 확장을 위한 자산입니다.

## 프레임워크를 쓰지 않은 이유

파이프라인이 분기 없는 3단계 직선이고 상태 전이가 없습니다. 프레임워크가 주는 추상화보다 호출 지점을 직접 보는 편이 디버깅에 유리합니다.

실제로 "왜 기각되는가"를 추적할 때 각 LLM 호출의 입출력을 그대로 볼 수 있어 위의 원인 A·B를 빠르게 찾았습니다. 프레임워크 내부에 감싸여 있었다면 프롬프트 문구 하나와 `temperature` 미설정이 원인이라는 걸 알아내기 더 어려웠을 것입니다.

LLM이 임의의 Cypher를 생성하게 하지 않습니다. 검색은 관계 유형·방향·깊이가 고정된 Cypher 템플릿만 씁니다.

## 실행

```bash
pip install -e .
cp .env.example .env    # OpenAI 키와 Neo4j 접속정보 입력

# 검색만 확인 (OpenAI는 임베딩 1회만 호출)
python scripts/query_retrieval.py "기대신용손실은 언제 인식하는가?"

# 답변까지 (임베딩 1회 + 생성 1회)
python scripts/ask.py "위험회피회계를 적용하기 위한 요건은?" --debug
```

`query_retrieval.py`는 각 청크의 `candidate_source`가 `hybrid`인지 `sibling`인지 표시하므로, 형제 보강이 실제로 무엇을 끌어왔는지 볼 수 있습니다.

필수 환경변수: `NEO4J_URI` · `NEO4J_USERNAME` · `NEO4J_PASSWORD` · `NEO4J_DATABASE` · `OPENAI_API_KEY` · `OPENAI_EMBEDDING_MODEL` · `OPENAI_CHAT_MODEL`

## 설정

`config/retrieval.yaml`

```yaml
hybrid:
  dense_top_k: 20        # Dense 벡터 검색 후보
  sparse_top_k: 20       # Sparse 전문 검색 후보
  seed_top_k: 12         # RRF 결합 후 상위
  rrf_k: 60
retrieval:
  top_k: 12              # 답변 생성기로 넘길 seed 수
  max_siblings: 8        # 형제 청크 보강 상한
```

`config/answering.yaml`

```yaml
answer:
  max_candidates: 20        # ★ top_k + max_siblings 이상이어야 함
  max_context_chars: 14000  # ★ 실질적 상한. 이 값이 먼저 걸린다
```

**`max_candidates`가 `top_k + max_siblings`보다 작으면 형제 청크가 잘려 나가 보강이 무의미해집니다.**

## 알려진 한계

- **인용 검증이 형식만 봅니다.** `evidence_id` 존재와 `citation` 일치는 확인하지만, 답변 문장이 근거 원문의 내용과 실제로 일치하는지는 검증하지 않습니다.
- **표현을 바꿔 인용하는 것은 허용됩니다.** 의미를 왜곡한 요약은 걸러지지 않습니다.
- **답변이 근거 범위를 넘는지 판단할 장치가 없습니다.** 이전의 의미 판정이 그 역할을 일부 했지만 기각률이 너무 높아 제거했습니다.
- **결과에 변동성이 있습니다.** 답변 생성 LLM에 `temperature`를 설정하지 않았습니다. 재정렬 제거로 변동 폭은 크게 줄었지만 0은 아닙니다.
- **정의 질문의 검색 편향.** 부록 정의표는 여러 용어를 한 Chunk에 담아, 해당 용어를 자주 언급하는 본문에 순위가 밀릴 수 있습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `src/accounting_rag/retrieval/` | Hybrid 검색, 형제 보강, 임베딩 |
| `src/accounting_rag/generation/` | 답변 생성과 결정적 검증 |
| `src/accounting_rag/ingestion/` | HWPX 파싱, 청킹, PDF 페이지 매핑 |
| `src/accounting_rag/graph/` | Neo4j 적재, 의미 KG |
| `src/accounting_rag/qa_pipeline.py` | 검색 → 생성 → 상태 판정 |
| `config/` | 검색·답변·그래프·청킹 설정 |
| `db/` | 제약조건·인덱스 DDL |
| `PROJECT_STRUCTURE.md` | 아키텍처, 설계 결정과 근거 |
| `NEO4J_SCHEMA.md` | 노드·관계·ID·인덱스 |

기준서 원문과 파생 데이터(파싱 결과·Chunk·임베딩)는 저작권 때문에 포함하지 않습니다.
