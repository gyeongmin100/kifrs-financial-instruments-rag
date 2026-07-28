# Rule·Condition·Exception·Example 의미 KG

공식 정의 기반 `Concept` 그래프와 별도로, 기준서 문단에서 규칙·조건·예외·예시 후보를 추출할 수 있다. OpenAI 출력은 항상 `candidate`로 적재되며 사람의 승인을 받기 전에는 GraphRAG 검색에 사용되지 않는다.

## 실행 순서

```powershell
# 1. 후보 추출. 먼저 소량을 검토한다.
python scripts/extract_semantic_candidates.py --standard-id 1109 --limit 50

# 2. 원문 인용, 원문 해시, Concept ID를 검증한다.
python scripts/validate_semantic_candidates.py

# 3. 승인하지 않은 상태로 Neo4j에 적재한다.
python scripts/load_semantic_candidates_neo4j.py
```

추출은 `data/semantic_candidates/extraction_cache.jsonl`에 배치별 결과를 기록하므로 중단 후 같은 설정으로 다시 실행할 수 있다. 모델이 반환한 `source_quote`가 원문에 정확히 존재하지 않으면 해당 후보는 적재 파일에서 제외된다. `applies_to_terms`는 기존의 승인된 공식 `Concept`와 정확히 대응되는 경우에만 `APPLIES_TO` 관계가 만들어진다.

전체 기준서 추출은 API 사용량이 발생한다. `--limit`으로 품질과 비용을 확인한 뒤 제한을 제거한다.

## 사람 검토

검토 화면은 공개 서비스에서 기본적으로 꺼져 있다. 로컬 `.env`에서 아래 두 값을 설정하고 API 서버를 다시 시작한 뒤 `/review`로 접속한다.

```dotenv
ENABLE_SEMANTIC_REVIEW=1
SEMANTIC_REVIEW_TOKEN=충분히_긴_임의의_토큰
```

토큰은 검토 페이지의 메모리에만 유지되며 브라우저 저장소에 기록하지 않는다. 승인 시 해당 후보의 `Rule`, 자식 의미 노드, `ESTABLISHES`, `HAS_CONDITION`, `HAS_EXCEPTION`, `APPLIES_TO`, `ILLUSTRATES` 관계가 함께 `approved`로 바뀐다. 거절 시 모두 `rejected`가 된다.

## 검색 안전 경계

GraphRAG는 다음 조건을 모두 만족하는 의미 경로만 탐색한다.

- 관계의 `review_status`가 `approved`이다.
- 대상이 `Rule`, `Condition`, `Exception`, `Example`이면 대상 노드도 `approved`이다.
- 의미 후보의 문장과 인용은 원문 문단 ID, 원문 해시, 기준서 번호와 함께 보존된다.

후보 적재를 반복해도 기존의 승인·거절 상태는 덮어쓰지 않는다.
