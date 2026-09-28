# Track 1: Scaling AI Agents

**Gemini Embedding 2 + Vector Search 2.0 으로 크로스모달 검색 엔진을 만들고, 실시간 쇼핑 에이전트로 배포합니다.**

-   https://github.com/cheeunlim/multimodal-agent

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/0.jpg)

---

## 진행 순서 (총 2시간)

| # | 구분 | 내용 |
| :--- | :--- | :--- |
| 1 | 환경 준비 | Qwiklabs 진입 → Gemini API Key 발급 (AI Studio) → `git clone` → `install.sh` (아래 절차) |
| 2 | 이론 1 | 크로스모달 임베딩 / Dense·Sparse·하이브리드 / RRF / 리랭킹 |
| 3 | **실습 1** | [`part1/`](part1/) — 청킹 → 임베딩 → 원리 검증 → Vector Search 2.0 검색·필터·리랭킹 |
| 4 | 배포 시작 | `gcloud run deploy` 실행 (백그라운드 빌드 진행 중 이론 2로 이동) |
| 5 | 이론 2 | kNN vs ANN(ScaNN), Live API 양방향 스트리밍, 에이전트 구조 |
| 6 | **실습 2** | [`part2/`](part2/) — 상품 컬렉션 질의: 텍스트 / 이미지 / RRF 가중치 / 리랭킹 → 앱 코드 대조 |
| 7 | 라이브 데모 | QR 생성 → 스마트폰 카메라·음성으로 에이전트 사용 |
| 8 | 마무리 | 자원 정리, Q&A |

> Cloud Run 소스 빌드에 약 5분이 소요되므로, **배포 명령을 먼저 실행해 두고 백그라운드 빌드가 진행되는 동안 이론 2를 진행**합니다. 이를 통해 실습 2를 시작할 때 이미 생성된 앱 URL을 바로 활용할 수 있습니다.

---

## 폴더 구조

```
multimodal-agent/
├── part1/                      # 실습 1 — 멀티모달 검색 원리 (노트북 + README)
├── part2/                      # 실습 2 — VS2 검색 엔진 + 에이전트 배포 (노트북 + app/ + README)
├── install.sh                  # 실습 환경 프로비저닝 스크립트
├── session2_index_builder.py   # Part 2 컬렉션·인덱스 빌더 (install.sh가 호출)
└── README.md                   # 이 문서
```

---

# 실습 준비 (약 10분)

## 1. Gemini API Key 발급 (Google AI Studio)

Part 2의 실시간 쇼핑 에이전트(Gemini Live API) 구동에 필요한 API Key를 발급받습니다.

#### 1-1. [Google AI Studio](https://aistudio.google.com/app/apikey) 에 접속합니다.
> Qwiklabs 실습용 계정(또는 개인 Google 계정)으로 로그인되어 있는지 확인합니다.

#### 1-2. **`Create API key`** (또는 `API 키 만들기`) 버튼을 클릭합니다.
* 프로젝트 선택 드롭다운에서 현재 실습 중인 GCP 프로젝트(또는 기본 프로젝트)를 선택하고 키를 생성합니다.

#### 1-3. 생성된 API Key를 복사하여 **메모장에 붙여 넣어 둡니다.**

> [!NOTE]
> 이 키는 **Part 2의 Cloud Run 배포에서만** 환경 변수(`GEMINI_API_KEY`)로 사용합니다.
> (Part 1 및 Part 2 노트북은 Workbench의 ADC 인증을 사용하므로 키 입력이 필요 없습니다.)

<br>

## 2. Workbench 실행 및 실습자료 다운로드

#### 2-1. 상단 검색 메뉴에서 `workbench` 를 입력해 'Workbench' 메뉴를 클릭합니다.
![image](https://raw.githubusercontent.com/jk1333/handson/main/images/6/1.png)

#### 2-2. `Open Jupyterlab` 버튼을 눌러 환경에 접속합니다.
![image](https://raw.githubusercontent.com/cheeunlim/agent-engine-lab/main/images/workbench_open.png)

#### 2-3. Terminal에 진입해 실습자료를 내려받습니다.

```bash
git clone https://github.com/cheeunlim/multimodal-agent
```

<br>

## 3. 환경 프로비저닝

```bash
cd ~/multimodal-agent
chmod +x ./install.sh
./install.sh
```

`install.sh` 가 수행하는 작업입니다.

| # | 작업 | 소요 |
| :--- | :--- | :--- |
| 0 | 필수 GCP API 활성화 (Vector Search, Vertex AI, Cloud Run 등) | ~1분 |
| 1 | 파이썬 패키지 설치 (`google-cloud-vectorsearch`, `google-genai` 등) | ~1분 |
| 2 | GCS 버킷 생성 (`gs://${PROJECT_ID}-vs2`, `asia-northeast1`) | 즉시 |
| 3 | Artifact Registry 리포 생성 (`cloud-run-source-deploy`, Part 2 배포용) | 즉시 |
| 4 | 상품 임베딩 데이터셋 복사 | ~10초 |
| 5 | **인덱스 빌더를 백그라운드로 구동** (`session2_index_builder.py`) | 임포트 ~10분, 이후 인덱스 |

> [!IMPORTANT]
> 5번 작업은 **백그라운드에서 비동기로 계속 진행**됩니다. 스크립트 실행이 완료된 후에도 인덱싱은 백그라운드에서 진행되며,
> 이론 1 및 실습 1 시간 동안 백그라운드에서 처리되므로 **바로 Part 1 실습을 시작하시면 됩니다.**

진행 상황은 아래로 볼 수 있습니다.

```bash
tail -f index_builder.log
```

## 실습 준비 완료!

<br>

---

# 실습

## [실습 1 — 멀티모달 검색 엔진의 원리](part1/) (25분)

비디오를 10초 단위로 분할하고, Gemini Embedding 2로 텍스트·이미지·비디오를 **하나의 3072차원 벡터 공간**에 임베딩합니다.
코사인 유사도 비교와 **비디오 궤적 + 카테고리 텍스트(`["빠름", "느림", "산", "바다"]`) 동시 t-SNE 투영**으로 멀티모달 공간을 직관적으로 확인한 뒤, 동일한 질의를 로컬 NumPy 완전탐색과 Vector Search 2.0에 각각 요청하여 결과를 대조하고 내장 RRF 하이브리드 검색·Ranking API 리랭킹까지 수행합니다.

## [실습 2 — VS2 검색 엔진과 실시간 쇼핑 에이전트](part2/) (25분 + 데모 10분)

약 10만 건(99,426건)의 Amazon 상품 컬렉션을 대상으로 텍스트 및 이미지 질의를 수행하고, RRF 가중치 변경에 따른 검색 순위 변화를
비교·분석합니다. 마지막으로 배포된 실시간 쇼핑 에이전트의 프롬프트와 툴 호출 흐름을 확인하고, 생성된 QR 코드(및 나노바나나 테스트 이미지)를 통해 스마트폰에서 에이전트를 직접 체험합니다.

> **진행 순서 안내**: `part2/README.md` 의 **Cloud Run 배포 명령을 먼저 실행**한 후 노트북 실습으로 이동합니다.

<br>

---

## 참고: 백그라운드 인덱싱 상태 점검 및 트러블슈팅

```bash
gcloud vector-search operations list --location=asia-northeast1 \
  --format='table(name.basename():label=OPERATION_ID, done, metadata.verb:label=STEP, error.message:label=ERROR)'
```

`install.sh` 는 총 **4개**의 장기 작업(LRO)을 만듭니다.

| # | 작업 | Part 2 시작 전 완료 필요? |
| :--- | :--- | :--- |
| 1 | 컬렉션 생성 (`.../collections/amazon-product-768-compact`) | ✅ **필수** (~10초) |
| 2 | 데이터 임포트 (`ImportDataObjectsMetadata`) | ✅ **필수** (~10분) |
| 3 | 텍스트 인덱스 생성 (`.../indexes/idx-text-embedding`) | 아니오 |
| 4 | 이미지 인덱스 생성 (`.../indexes/idx-image-embedding`) | 아니오 |

**1번과 2번 작업이 `done: True` 상태가 되면 실습 2를 바로 진행할 수 있습니다** (약 10분 소요).
인덱스(3·4번) 생성이 진행 중이더라도 검색은 **kNN 완전탐색** 방식으로 정상 동작합니다.

> [!NOTE]
> **인덱스는 컬렉션당 하나씩 순차적으로 생성됩니다.** 3번 작업이 완료된 후 4번 작업이 요청되므로,
> 진행 중에는 목록에 LRO 작업이 **최대 3개까지만** 표시되는 것이 정상입니다.

---

## 🔧 자주 발생하는 에러 & 트러블슈팅 가이드

### 1. 에러 로그 확인 명령어 모음

| 상황 | 확인 명령어 |
| :--- | :--- |
| **백그라운드 인덱스 빌더 로그 확인** | `tail -n 30 ~/multimodal-agent/index_builder.log` |
| **Vector Search LRO 작업 상세/에러 조회** | `gcloud vector-search operations describe <OPERATION_ID> --location=asia-northeast1` |
| **Cloud Run 에이전트 서버 로그 확인** | `gcloud run services logs read lens-mosaic --region=asia-northeast1 --limit=50` |
| **현재 활성화된 계정 및 프로젝트 확인** | `gcloud auth list` 및 `gcloud config get-value project` |

### 2. `409 Conflict` (`AlreadyExists` / `Aborted`) 에러가 발생할 때

*   **증상 1 (`409 AlreadyExists`)**: 컬렉션 생성 셀이나 `install.sh`를 두 번 실행했을 때 리소스가 이미 존재하여 발생합니다.
    *   **대처**: 실습 코드와 `session2_index_builder.py` 모두 `409 AlreadyExists`를 자동 감지해 **기존 컬렉션/데이터를 그대로 재사용**하도록 처리되어 있으므로 에러가 아니며 그대로 다음 단계를 진행하시면 됩니다.
*   **증상 2 (`409 Aborted: unable to queue the operation`)**: 컬렉션 하나에는 한 번에 하나의 인덱스 생성 작업만 큐에 들어갈 수 있습니다. 앞선 작업(예: `text_embedding` 인덱스)이 진행 중일 때 다음 작업이 요청되면 발생합니다.
    *   **대처**: `session2_index_builder.py`가 앞 인덱스 빌드가 끝날 때까지 2분 간격으로 자동 대기 후 재요청하므로 무시하셔도 됩니다.
*   **증상 3 (`install.sh` 두 번 실행으로 인한 중복 임포트 우려)**:
    *   **대처**: `install.sh`는 이미 실행 중인 `session2_index_builder.py` 프로세스가 있으면 중복 실행을 차단하며, 빌더 내부에서도 적재 건수(`COUNT > 0`)를 먼저 확인해 중복 임포트를 방지합니다.

### 3. 계정 꼬임 (`403 PermissionDenied` / 프로젝트 불일치 / 에이전트 접속 끊김)

*   **Workbench / 터미널 프로젝트 불일치**:
    *   `install.sh` 실행 시 출력되는 `Active Account`와 `Project ID`가 Qwiklabs 실습 계정(`student-...` / `qwiklabs-gcp-...`)이 맞는지 확인하세요.
    *   만약 프로젝트가 비어 있거나 다른 프로젝트로 잡혀 있다면 아래 명령으로 고정합니다:
        ```bash
        gcloud config set project <실습_PROJECT_ID>
        ```
*   **모바일 에이전트 접속 시 `Disconnected - reconnecting...` 반복 (WebSocket 1008 에러)**:
    *   **원인 1**: Cloud Run 배포 시 `--set-env-vars GEMINI_API_KEY="..."` 자리에 실제 키 대신 플레이스홀더(`-----GEMINI_API_KEY-----`)가 그대로 들어갔거나 공백/따옴표 오타가 있는 경우.
    *   **원인 2**: AI Studio에서 API Key를 만들 때 브라우저가 **개인 Gmail 계정**으로 로그인되어 있어, API가 비활성화된 엉뚱한 프로젝트의 키가 발급된 경우.
    *   **해결**: 시크릿 창에서 **Qwiklabs 학생 계정**으로 [Google AI Studio](https://aistudio.google.com/app/apikey)에 접속해 현재 실습 프로젝트(`qwiklabs-gcp-...`)의 API Key를 새로 발급받은 뒤, `gcloud run deploy` 명령을 다시 실행하세요.
