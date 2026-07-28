# Neo4j 그래프 스키마

## 1. 목적과 적용 범위

이 문서는 K-IFRS 제1032호·제1039호·제1107호·제1109호의 파싱 결과와 검색용 Chunk를 Neo4j 5.x에 적재하기 위한 기준이다. 선언형 벡터 인덱스 구문을 그대로 실행할 수 있도록 최소 Neo4j 5.15를 기준으로 한다. Neo4j 하나에 문서 구조 그래프, 명시적 참조 그래프, 전문 검색 인덱스, 벡터 인덱스를 함께 둔다.

스키마의 핵심 원칙은 다음과 같다.

- 원문에서 기계적으로 확인되는 사실과 OpenAI가 추론한 의미 관계를 섞지 않는다.
- 모든 노드에는 재실행해도 바뀌지 않는 고유 ID를 부여한다.
- 모든 관계에는 `provenance`, `confidence`, `review_status`를 저장한다.
- Chunk는 원문 노드를 대체하지 않는 검색용 파생 데이터다.
- 승인되지 않은 의미 관계는 기본 검색 경로에 포함하지 않는다.

실행 가능한 설정은 [config/graph_schema.yaml](config/graph_schema.yaml), Neo4j DDL은 [db/schema.cypher](db/schema.cypher)에 있다.

## 2. 실제 입력 데이터

| 입력 | 실제 핵심 필드 | 그래프 사용처 |
|---|---|---|
| `paragraphs.jsonl` | `paragraph_id`, `number`, `zone`, `section_path`, `text`, `subitems`, `document_order`, 페이지 매핑 필드 | Paragraph, Subparagraph, Section, Zone |
| `blocks.jsonl` | `block_id`, `block_type`, `parent_paragraph_id`, `table_ids`, `footnote_ids`, `references` | Block와 원문 부속 관계 |
| `tables.jsonl` | `table_id`, `parent_paragraph_id`, `cells`, 행·열, 페이지 | Table |
| `footnotes.jsonl` | `footnote_id`, `parent_paragraph_id`, `number`, `text`, 페이지 | Footnote |
| `pdf_pages.jsonl` | `standard_id`, `pdf_page`, `pdf_page_label`, `text` | PdfPage |
| `references.jsonl` | 출발점, 종류, 문자 위치, 대상 범위, 해석 상태, `resolved_target_ids` | REFERS_TO |
| `chunks.jsonl` | `chunk_id`, 검색 텍스트, 모든 원문 ID, 검색 여부, 페이지, 우선순위 | Chunk와 DERIVED_FROM |

현재 참조 데이터에는 단일 문단, 문단 범위, 하위항목, 하위항목 범위, 다른 기준서 참조가 모두 존재한다. `resolved`, `resolved_range`, `resolved_standard`, `external_standard`만 A등급 관계로 자동 적재한다. `unresolved_*`, `ambiguous*`는 검토 데이터로 보존하되 `REFERS_TO`로 확정하지 않는다.

## 3. 신뢰 등급

### A등급: 자동 적재 가능

파서·청크 생성기·페이지 매퍼·명시 참조 파서가 원문 구조 또는 명확한 표기를 바탕으로 만든 사실이다.

- 문서 계층과 순서
- 문단과 하위항목
- Block, Table, Footnote, PDF 페이지
- Chunk의 원문 출처
- 해석이 완료된 명시적 문단·범위·기준서 참조

기본값은 `confidence=1.0`, `review_status='approved'`다. 페이지처럼 알고리즘 신뢰도가 별도로 있는 경우 실제 `page_match_confidence`를 사용한다.

### B등급: LLM 추론이 개입한 의미 관계

LLM이 원문을 해석해 제안한 의미 관계다. 후보 생성 직후에는 `review_status='candidate'`이며 사람이 원문 문구·출처 문단·방향을 검증해 `approved`가 되기 전까지 기본 검색과 답변 근거에서 제외한다.

**현재 적재된 B등급 노드·관계는 없다.** 규칙·조건·예외·사례 추출을 검토했으나 승인 절차를 운영할 계획이 없어 최종 범위에서 제외했고, 관련 정의도 스키마에서 삭제했다.

`Concept`는 LLM 추론이 아니라 공식 용어 정의표에서 결정적으로 생성하므로 A등급이다.

### C등급: 미확정 후보

출처 문구가 없거나 대상·방향이 모호한 후보다. 그래프 본 테이블에 자동 병합하지 않고 별도 검토 JSONL에 보관한다.

## 4. 노드

### 4.1 A등급 구조 노드

| Label | 고유 키 | 설명 |
|---|---|---|
| `Standard` | `standard_id` | 적재된 K-IFRS 기준서 |
| `Zone` | `zone_id` | 본문, 적용지침, 결론도출근거 등 문서 영역 |
| `Section` | `section_id` | `section_path`의 각 단계 |
| `Paragraph` | `paragraph_id` | 공식 문단번호가 있는 문단 |
| `Subparagraph` | `subparagraph_id` | 문단 내부의 ⑴, ㈎ 등 번호 단위 |
| `Block` | `block_id` | HWPX 원본 순서를 보존하는 최소 블록 |
| `Table` | `table_id` | 셀 구조와 직렬화 텍스트를 가진 표 |
| `Footnote` | `footnote_id` | 각주 번호와 본문 |
| `PdfPage` | `page_id` | 기준서별 PDF 실제 페이지 |
| `Chunk` | `chunk_id` | Dense·Sparse 검색용 파생 텍스트 |
| `ExternalStandard` | `external_standard_id` | 참조되었지만 현재 네 기준서에는 적재되지 않은 기준서 |

Standard의 `standard_id`는 원본 값인 `1032`, `1039`, `1107`, `1109`를 그대로 사용한다. Section ID는 정규화한 전체 경로의 SHA-256 앞 16자리로 만든다. 제목만 사용하면 서로 다른 위치의 같은 제목이 충돌하기 때문이다. PdfPage ID는 `KIFRS{기준서}-PAGE-{4자리 페이지}`다.

### 4.2 의미 노드

| Label | 의미 |
|---|---|
| `Concept` | 금융자산, 기대신용손실 등 공식 정의에서 정규화한 A등급 회계 개념 |

의미 노드는 `Concept` 하나만 사용한다. `extractor_model`, `extractor_version`, `review_status`를 갖는다. Concept는 제1107·1109호의 정의표, 제1032호 문단 11, 제1039호 문단 9만 원천으로 사용한다. 동일한 공식 용어만 공유 Concept로 병합하고, 원문에 명시된 alias만 저장한다.

## 5. 관계와 방향

```text
(Standard)-[:CONTAINS]->(Zone)
(Standard)-[:CONTAINS]->(PdfPage)
(Zone)-[:CONTAINS]->(Section)
(Section)-[:CONTAINS]->(Section|Paragraph)
(Paragraph)-[:CONTAINS]->(Subparagraph)
(Subparagraph)-[:CONTAINS]->(Subparagraph)

(Paragraph)-[:HAS_BLOCK]->(Block)
(Section)-[:HAS_BLOCK]->(Block)       // 부모 문단이 없는 블록
(Paragraph|Block)-[:HAS_TABLE]->(Table)
(Paragraph|Block)-[:HAS_FOOTNOTE]->(Footnote)
(Paragraph|Block|Table|Footnote|Chunk)-[:APPEARS_ON]->(PdfPage)
(Chunk)-[:DERIVED_FROM]->(Paragraph|Subparagraph|Block|Table)

(Paragraph|Block|Table|Footnote)-[:REFERS_TO]->(Paragraph|Subparagraph|Standard|ExternalStandard)
```

`NEXT`는 문서의 앞에서 뒤 방향으로 한 번만 저장한다. 탐색할 때는 양방향으로 읽어 Seed의 앞·뒤 한 개씩, 즉 ±1만 가져온다. 범위 참조는 범위에 실제로 포함되는 각 대상 문단으로 개별 `REFERS_TO`를 만들고, 동일한 `reference_group_id`로 묶는다.

의미 관계는 다음과 같다.

```text
(Paragraph|Block|Table)-[:MENTIONS]->(Concept)
```

`MENTIONS`는 공식 정의의 canonical name 또는 승인된 alias가 원문에 정확히 등장할 때만 만든다. 문자열 일치로 결정되므로 A등급이다.

관계 방향은 의미가 읽히는 방향으로 고정한다. 예를 들어 예외가 규칙을 가리키는 것이 아니라 규칙이 예외를 가진다.

## 6. 관계 공통 속성

모든 관계에 최소 다음 속성을 둔다.

| 속성 | 설명 |
|---|---|
| `provenance` | `parser`, `reference_parser`, `chunk_builder`, `page_mapper`, `openai` 등 생성 주체 |
| `confidence` | 0~1 신뢰도 |
| `review_status` | `approved`, `candidate`, `rejected` |

가능한 경우 `source_id`, `source_paragraph_id`, `source_text_span`, `method`, `reference_id`, `reference_group_id`, `extractor_model`, `extractor_version`도 저장한다. 의미 관계에는 원문 추적을 위해 `source_paragraph_id`와 정확한 `source_text_span`이 필수다.

## 7. 탐색 정책

| 탐색 종류 | 정책 | 이유 |
|---|---|---|
| 의미·참조 관계 | 기본 2-hop, 근거 부족 시 최대 3-hop | 조건·예외·참조까지 도달하되 무관한 확장을 제한 |
| 계층 관계 | 최대 6-hop | Section 중첩 깊이가 문서마다 달라도 Standard부터 하위 요소까지 도달 |
| 인접 문맥 | `NEXT` 양방향 ±1 | 앞뒤 문맥은 보강하지만 긴 문서 전체로 확산하지 않음 |

의미 탐색에서는 반드시 `review_status='approved'`인 관계만 사용한다. A등급 참조도 해석 상태가 확정된 것만 적재한다. hop 수는 관계의 edge 개수이며, 계층 hop과 의미·참조 hop 예산은 별도로 계산한다.

## 8. 인덱스

`db/schema.cypher`는 다음을 재실행 가능하게 생성한다. Concept 이름·별칭·정의에는 `concept_fulltext` CJK 전문 인덱스를 사용한다.

- 모든 노드 ID의 고유 제약조건
- 기준서·문단번호, Zone·순서, 페이지, Chunk 필터용 조회 인덱스
- Chunk의 `text`, `contextualized_text`, `citation_label` 전문 인덱스
- Chunk `embedding` 벡터 인덱스

OpenAI `text-embedding-3-large`의 기본 차원인 3,072를 사용한다. `config/graph_schema.yaml`과 Cypher의 차원은 항상 같아야 한다. 나중에 임베딩 차원을 바꾸면 기존 벡터를 전부 다시 생성하고 `chunk_embedding_vector` 인덱스를 삭제 후 새 차원으로 재생성해야 한다. 다른 차원의 벡터를 기존 인덱스에 섞어 넣으면 안 된다.

전문 검색은 한국어 복합어를 bi-gram으로 색인하는 Neo4j `cjk` 분석기를 사용한다. `standard-no-stop-words`는 한국어 질의에서 관련 없는 결과를 반환하는 것이 실제 검증되어 사용하지 않는다. 검색 결과에는 애플리케이션 쿼리에서 `searchable=true AND inactive=false`를 적용한다. Neo4j 전문 인덱스 자체는 이 조건을 강제하지 않는다.

## 9. 적재 순서

1. A등급 구조 노드와 관계를 `MERGE`로 적재한다.
2. 해석 완료된 명시 참조만 `REFERS_TO`로 적재한다.
3. Chunk 임베딩을 생성해 `Chunk.embedding`에 저장한다.
4. `scripts/build_semantic_kg.py`로 공식 정의 Concept·MENTIONS JSONL을 만들고 `scripts/load_semantic_neo4j.py`로 적재한다.
5. `scripts/validate_semantic_kg.py`로 개수·정의 출처·승인 상태·인덱스·실제 2-hop 정의 경로를 확인한다.

적재기는 재실행해도 중복 노드·관계를 만들지 않는다. 현재 적재된 의미 데이터는 공식 정의 기반 `Concept`와 `MENTIONS`뿐이며, LLM 추론으로 만든 B등급 노드·관계는 없다.
