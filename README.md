# K-IFRS Financial Instruments RAG — Hybrid RAG + GraphRAG

K-IFRS 금융상품 기준서 4개(제1032호·제1039호·제1107호·제1109호)를 구조화하여 **Neo4j 하나**에 적재하고, **Hybrid 검색(Dense + Sparse) → GraphRAG 확장 → LLM 재정렬 → 근거 충분성 검사 → 인용 검증 답변**의 직렬 파이프라인으로 기준서 원문을 근거로 답변하는 질의응답 시스템입니다.

```text
사용자 질문
  → 검색용 질문 분석 (기준서·문단·개념 추출, OpenAI Structured Output)
  → Dense Vector 검색 (text-embedding-3-large, Neo4j Vector Index)
  → Sparse 검색 (Neo4j Full-text Index, CJK analyzer)
  → weighted RRF로 Hybrid 결합
  → Hybrid 상위 결과를 Seed로 Neo4j 그래프 확장 (기본 1-hop, 조건부 2-hop)
  → 후보 통합 → OpenAI Structured Output 재정렬
  → 근거 충분성 검사 (결정적 + 의미적) → 부족 시 최대 1회 보완 검색
  → 결론 / 판단 과정 / 근거 형식 답변 생성
  → 인용 ID·기준서 문단 무결성 검증
```

## 왜 이 구조인가

병렬 Hybrid RAG + GraphRAG가 아닌 **직렬 구조**(Hybrid 결과를 그래프 탐색의 시작점으로 사용)를 선택했습니다.

- 질문과 무관한 그래프 노드에서 탐색이 시작되는 것을 방지
- 전체 그래프에서 엔티티를 먼저 찾는 방식보다 안정적 (엔티티 연결 실패에 강함)
- 검색 결과와 그래프 경로를 완전히 역추적 가능
- 그래프 확산과 토큰 사용량을 명시적으로 제한
- 회계기준서처럼 정확한 문단 인용과 국소적 참조 관계가 중요한 도메인에 적합

## 핵심 구현 포인트

### 1. 원문 구조를 보존하는 HWPX 파싱

- 문단·하위항목·표·각주·문서 영역(본문/부록/적용지침/적용사례 등)을 계층 구조로 보존
- 표는 평탄화하지 않고 행·열·셀 주소·병합 정보 + 검색용 직렬화 텍스트를 함께 유지
- 문단 번호를 기본 인용 단위로 유지, 긴 문단은 하위항목·소분류 등 구조 경계에서만 분할
- 삭제·시행일·경과규정은 별도 속성으로 보존, 앞부분(표지·목차 등)은 보존하되 검색 우선순위 메타데이터로 제어

### 2. 명시적 참조 그래프 (Reference Graph)

- 단일·복수·범위·하위항목·타 기준서 참조를 파싱 (`문단 3.2.3~3.2.9` 등 범위 기호 5종 지원)
- 범위 참조는 숫자 계산이 아니라 **해당 기준서에 실제 존재하는 문단의 문서 순서**로 확장
- 각 대상 문단에 개별 `REFERS_TO` 관계 생성, 같은 범위는 `reference_group_id`로 묶음
- 확정할 수 없는 참조(삭제된 문단, 미적재 기준서 등)는 억지로 연결하지 않고 `unresolved`/`ambiguous` 상태로 보존

### 3. 공식 정의 기반 Concept KG

- 기준서의 공식 용어 정의(제1107·1109호 정의표, 제1032호 문단 11, 제1039호 문단 9)에서 Concept 45개를 결정적으로 생성
- LLM의 "의미가 비슷하다" 판단으로 병합하지 않고 **동일한 공식 용어일 때만** 공유 Concept로 병합
- 본문·표·적용사례에서 canonical name/승인된 alias가 정확히 등장할 때만 `MENTIONS` 연결 (17,712개)
- Concept는 탐색용 중간 노드이며 최종 인용 근거로는 사용하지 않음 — 근거는 항상 기준서 원문

### 4. PDF 페이지 매핑

- HWPX가 구조의 기준, PDF(1,671쪽)가 페이지 인용의 기준
- 검색 대상 문단 3,891개·Chunk 4,144개 전체에 PDF 페이지 부여 (exact/fuzzy/inferred/inherited 매핑 상태 관리)
- 저신뢰·복수 후보는 검토 목록으로 분리하여 감사 추적 유지

### 5. 제한된 GraphRAG 탐색

- `Chunk → DERIVED_FROM → 원문 노드` 복원은 hop으로 계산하지 않음
- 기본 1-hop, 명시적 참조·부모 복원·질문어 일치 시에만 최대 2-hop
- 결과 최대 40개, 2-hop frontier 최대 12개, 방문 집합으로 순환 방지
- 관계 화이트리스트: `REFERS_TO`, `MENTIONS`, `CONTAINS`, `HAS_BLOCK`, `HAS_TABLE`, `HAS_FOOTNOTE`, `NEXT`
- 문서 전체로 확산 가능한 관계(`APPEARS_ON` 등)는 의도적으로 제외

### 6. Structured Output 재정렬과 안전장치

- OpenAI Responses API strict JSON Schema로 후보별 채점: 질문 관련성 40% / 본문 우선도 20% / 직접 근거성 30% / 참조 연결성 10%
- API 실패·스키마 위반·알 수 없는 ID 발생 시 입력 순서를 보존하는 결정적 fallback
- 답변 전 근거 충분성 검사(결정적 조건 + LLM 의미 판단), `missing_aspects`가 명확할 때만 최대 1회 보완 검색
- 답변 후 인용 검증: 존재하지 않는 문단 인용, 후보 밖 근거 사용을 차단. 검증 실패 시 답변 미노출
- 근거가 부족하면 추측하지 않고 `insufficient` 반환

## 데이터 규모

| 항목 | 수치 |
|---|---|
| 문단 (공식 번호) | 3,904 |
| 원문 블록 | 8,533 |
| 표 / 각주 | 362 / 227 |
| 명시적 참조 | 7,450 |
| 검색용 Chunk | 4,379 (검색 대상 4,144, 전체 임베딩 보유) |
| Neo4j 관계 | 약 74,409 (+ MENTIONS 17,712) |
| Concept | 45 |
| PDF 페이지 매핑 | 1,671쪽 기준, 검색 대상 Chunk 미매핑 0개 |

## 기술 스택

- Python 3.11+ / Neo4j 5.x (단일 DB — 구조 그래프 + Vector Index + CJK Full-text Index)
- OpenAI API (질문 분석·재정렬·충분성 검사·답변 생성: Structured Output, 임베딩: text-embedding-3-large 3,072차원)
- 별도 벡터 DB·Elasticsearch·LangChain 오케스트레이션 없이 명시적 Cypher와 일반 Python 파이프라인으로 구현

## 저장소 범위

이 저장소는 **RAG 구현 코드·스키마·설정·문서만** 포함합니다.

포함하지 않는 것:

- K-IFRS 기준서 원본 (HWP/HWPX/PDF) — 저작권상 재배포 범위 미확정
- 파싱 결과·Chunk·임베딩·의미 KG 등 원문 파생 데이터 (`data/` 산출물)
- `.env` (API 키·DB 접속정보)
- 웹 UI·API 서빙 계층 (로컬 별도 구현, 저장소 범위 제외)

따라서 저장소만으로는 실행할 수 없으며, 기준서 원본 파일을 보유한 환경에서 아래 파이프라인을 순서대로 실행해야 합니다.

## 파이프라인 실행 (원본 보유 환경 기준)

```powershell
pip install -e .
Copy-Item .env.example .env   # OpenAI 키와 Neo4j 접속정보 입력

# 1. 파싱 → 검증
python scripts/parse_all_standards.py
python scripts/validate_processed_dataset.py data/processed

# 2. PDF 페이지 매핑
python scripts/map_pdf_pages.py
python scripts/validate_page_mapping.py

# 3. 검색용 Chunk 생성
python scripts/build_chunks.py
python scripts/validate_chunks.py

# 4. Neo4j 구조·참조 그래프 적재
python scripts/load_neo4j.py
python scripts/validate_neo4j.py

# 5. 의미 KG (Concept + MENTIONS)
python scripts/build_semantic_kg.py
python scripts/validate_semantic_kg.py
python scripts/load_semantic_neo4j.py

# 6. 임베딩 생성·적재
python scripts/build_embeddings.py
python scripts/load_embeddings_neo4j.py
python scripts/validate_embeddings.py

# 7. 검색·질의응답
python scripts/query_hybrid.py "기대신용손실은 언제 인식하는가?"
python scripts/query_retrieval.py "기대신용손실 측정 시 미래전망정보를 어떻게 반영하는가?"
python scripts/ask.py "기대신용손실 측정 시 미래전망정보를 어떻게 반영하는가?" --debug
```

## 프로젝트 구조

```text
├─ config/                  # 검색·청킹·임베딩·그래프 스키마 설정
├─ db/
│  ├─ schema.cypher         # 제약조건, 조회·전문(CJK)·벡터 인덱스
│  └─ migrations/
├─ scripts/                 # 파싱→매핑→청킹→적재→검색→질의응답 CLI
├─ src/accounting_rag/
│  ├─ ingestion/            # HWPX 파서, 데이터셋 빌더, PDF 매퍼, Chunk 빌더
│  ├─ graph/                # Neo4j 적재기, 의미 KG 빌더·적재기
│  ├─ retrieval/            # 임베딩, Hybrid(RRF), Graph 확장, 재정렬, 파이프라인
│  ├─ query/                # 검색용 질문 분석 (Structured Output)
│  ├─ generation/           # 충분성 검사, 답변 생성, 인용 검증
│  └─ qa_pipeline.py        # 전체 질의응답 오케스트레이션
└─ tests/                   # 회귀 테스트 (참고용)
```

## 상세 문서

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) — 설계 결정, 파싱·청킹 정책, 검색 파이프라인 상세
- [NEO4J_SCHEMA.md](NEO4J_SCHEMA.md) — 노드·관계·신뢰 등급·탐색 정책
- [PROJECT_WORKFLOW.html](PROJECT_WORKFLOW.html) — 단계별 구현 이력과 설계 근거
