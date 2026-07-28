# 배포 가이드

현재 서비스는 아래 무료 구성으로 운영한다.

```text
사용자 브라우저
    │
    ▼
Cloudflare Pages  ← 프론트엔드 (정적 HTML/CSS/JS)
    │  POST /v1/jobs → GET /v1/jobs/{id} 폴링
    ▼
Render 무료 웹 서비스  ← 백엔드 (FastAPI, Docker)
    │                     · CORS 허용 목록, IP당 요청 제한
    ├────────────► Neo4j AuraDB Free  ← 그래프 + 벡터 + 전문 검색
    └────────────► OpenAI API         ← 임베딩·답변 생성·이미지 판독
```

Hugging Face Spaces를 먼저 검토했으나 Docker Space가 PRO 구독 전용으로 바뀌어 Render로 변경했다.

## 사전 준비

| 서비스 | 주소 | 필요한 것 |
|---|---|---|
| Neo4j Aura | https://console.neo4j.io | AuraDB Free 인스턴스, 생성 시 자격증명 파일 다운로드 |
| GitHub | https://github.com | 배포용 저장소(비공개 가능) |
| Render | https://render.com | GitHub 로그인, 저장소 접근 권한 |
| Cloudflare | https://dash.cloudflare.com | Pages 사용 |

전부 무료이며 신용카드가 필요 없다.

### Aura Free 주의사항

- 인스턴스 생성 시 표시되는 **비밀번호는 그 순간에만 보여준다.** 자격증명 파일을 반드시 저장한다.
- **username과 database 이름이 `neo4j`가 아니라 인스턴스 ID다.** 자격증명 파일의 `NEO4J_USERNAME`, `NEO4J_DATABASE` 값을 그대로 사용해야 한다. 이 값을 `neo4j`로 두면 인증 실패(`Neo.ClientError.Security.Unauthorized`)가 난다.
- 콘솔 SSO 사용자에게는 `SHOW USERS`, `ALTER USER` 권한이 없다. 비밀번호를 잃으면 인스턴스를 삭제 후 재생성하는 편이 빠르다.
- **3일 미사용 시 자동 일시정지** → 콘솔에서 Resume 필요. 백엔드가 503을 반환하면 먼저 이 상태를 확인한다.
- **30일 정지 지속 시 삭제.** 로컬 데이터가 원본이므로 재적재로 복구한다.
- 용량 제한은 노드 200,000 / 관계 400,000이며 현재 규모(약 22,000 / 92,000)는 여유가 있다.

## 1단계. Aura에 데이터 적재 (로컬에서 실행)

`.env`의 로컬 Neo4j 설정은 그대로 두고, PowerShell 세션 환경변수로 Aura를 지정해 적재한다.
`load_dotenv`는 이미 설정된 환경변수를 덮어쓰지 않으므로 세션 값이 우선한다.

```powershell
# 값은 모두 Aura 자격증명 파일(Neo4j-<ID>-Created-*.txt) 기준
$env:NEO4J_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"
$env:NEO4J_USERNAME = "<NEO4J_USERNAME (보통 인스턴스 ID)>"
$env:NEO4J_PASSWORD = "<NEO4J_PASSWORD>"
$env:NEO4J_DATABASE = "<NEO4J_DATABASE (보통 인스턴스 ID)>"

python scripts/load_neo4j.py            # 구조 그래프 + 제약조건·인덱스
python scripts/load_semantic_neo4j.py   # Concept + MENTIONS (현재 질의에는 미사용)
python scripts/load_embeddings_neo4j.py # 임베딩 (캐시 재사용, OpenAI 호출 없음)

python scripts/validate_neo4j.py
python scripts/validate_semantic_kg.py
python scripts/validate_embeddings.py
```

적재가 끝나면 터미널을 닫거나 `Remove-Item Env:NEO4J_*`로 원복한다.

기대 결과: 노드 22,018개, 관계 92,121개, 검색 대상 Chunk 4,144개 전체 임베딩 보유,
`chunk_embedding_vector`와 `chunk_fulltext`(analyzer=`cjk`) 모두 ONLINE.

## 2단계. 배포용 저장소 준비

Render는 GitHub 저장소에서 코드를 가져온다. 비공개 저장소도 된다.

포함할 것: `Dockerfile`, `pyproject.toml`, `src/`, `config/`
제외할 것: 기준서 원본, `data/` 산출물, `.env`

```powershell
git init -b main
git add Dockerfile pyproject.toml src config
git commit -m "Add backend for deployment"
gh repo create accounting-rag-deploy --private --source . --push
```

GitHub에서 Render 앱에 저장소 접근을 허용해야 한다. 대시보드에서 연결되지 않으면
https://github.com/apps/render/installations/new 에서 직접 설치한다.

## 3단계. Render 웹 서비스 생성

New → Web Service → 저장소 선택 후 다음을 지정한다.

- Runtime: **Docker**
- Plan: **Free**
- Dockerfile Path: `./Dockerfile`

환경변수는 다음과 같이 등록한다. 키·비밀번호는 Secret으로 저장한다.

| 이름 | 값 | 구분 |
|---|---|---|
| `OPENAI_API_KEY` | 실제 키 | Secret |
| `OPENAI_CHAT_MODEL` | gpt-5.6-terra | 일반 |
| `OPENAI_EMBEDDING_MODEL` | text-embedding-3-large | 일반 |
| `NEO4J_URI` | neo4j+s://xxxx.databases.neo4j.io | Secret |
| `NEO4J_USERNAME` | 자격증명 파일 값 | 일반 |
| `NEO4J_PASSWORD` | 자격증명 파일 값 | Secret |
| `NEO4J_DATABASE` | 자격증명 파일 값 | 일반 |
| `CORS_ALLOW_ORIGINS` | https://<프로젝트>.pages.dev | 일반 |
| `ASK_RATE_LIMIT_PER_HOUR` | 10 | 일반 |
| `TRUST_PROXY_HEADERS` | 1 | 일반 |

`CORS_ALLOW_ORIGINS`는 5단계에서 Pages 주소가 정해진 뒤 등록해도 된다.

빌드 완료 후 `https://<서비스명>.onrender.com/health`가 `{"status":"ok"}`를 반환하는지 확인한다.

## 4단계. 프론트엔드 번들 생성

```powershell
python scripts/build_frontend_dist.py --api-base https://<서비스명>.onrender.com
```

`deploy/frontend_dist/`에 `index.html`과 `static/`이 생성되고 `config.js`에 백엔드 주소가 주입된다.

## 5단계. Cloudflare Pages 배포

1. dash.cloudflare.com → Workers & Pages → Create → Pages → **Upload assets**
2. 프로젝트 이름 입력 후 `deploy/frontend_dist/` 업로드 → Deploy
3. 발급된 `https://<프로젝트>.pages.dev` 주소를 확인
4. 그 주소를 Render의 `CORS_ALLOW_ORIGINS`에 등록 (서비스가 자동 재시작된다)

ZIP으로 올릴 경우 경로 구분자가 `/`인지 확인한다. Windows에서 `Compress-Archive`로 만들면
`static\app.js`처럼 역슬래시가 들어가 폴더 구조가 깨질 수 있다.

## 6단계. 확인

```powershell
# 백엔드 단독
curl https://<서비스명>.onrender.com/health

# 종단 (프론트 origin 기준 CORS 포함)
python -c "import httpx,time; API='https://<서비스명>.onrender.com'; r=httpx.post(f'{API}/v1/jobs', json={'question':'기대신용손실 측정 시 미래전망정보를 어떻게 반영하는가?','top_k':10}, timeout=60); j=r.json()['job_id']; print(r.status_code); [time.sleep(6) for _ in range(1)]"
```

브라우저에서 `https://<프로젝트>.pages.dev`를 열고 질문해 답변과 근거가 표시되는지 확인한다.

## 재배포

- **백엔드**: 배포 저장소에 push하면 Render가 자동 빌드한다.
- **프론트엔드**: 4단계로 번들을 다시 만들고 Pages에 새 배포로 업로드한다.

## 운영 메모

- **비용**: 인프라는 전액 무료. OpenAI API만 질문당 과금된다. IP당 시간당 10회 제한이 걸려 있다.
- **콜드 스타트**: Render 무료 플랜은 15분 유휴 후 슬립하여 첫 요청이 최대 1분가량 걸린다. 비동기 job 방식이라 브라우저가 끊겨도 결과를 잃지 않는다.
- **Aura 정지**: 3일 미사용 시 콘솔에서 Resume가 필요하다.
- **로컬 실행은 그대로**: `python scripts/run_api.py`. 배포용 환경변수를 설정하지 않으면 CORS와 요청 제한이 비활성이라 기존과 동일하게 동작한다.

## 배포 전용 환경변수 (모두 선택)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | (없음) | 허용할 프론트엔드 origin, 쉼표 구분. 미설정 시 CORS 헤더를 보내지 않는다 |
| `ASK_RATE_LIMIT_PER_HOUR` | 0 | IP당 시간당 요청 허용 횟수. 0이면 무제한 |
| `TRUST_PROXY_HEADERS` | (없음) | `1`이면 `X-Forwarded-For`에서 클라이언트 IP를 읽는다. 리버스 프록시 뒤에서만 사용 |
