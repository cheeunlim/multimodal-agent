# Track 1: Scaling AI Agents

**Gemini Embedding 2 + Vector Search 2.0 으로 크로스모달 검색 엔진을 만들고, 실시간 쇼핑 에이전트로 배포합니다.**

-   https://github.com/cheeunlim/smx-multimodal-agent

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/0.jpg)

---

## 진행 순서 (총 2시간)

| 시간 | 구분 | 내용 |
| :--- | :--- | :--- |
| 00~10분 | 환경 준비 | Qwiklabs 진입 → API Key 확인 → `git clone` → `install.sh` (아래 절차) |
| 10~30분 | 이론 1 | 크로스모달 임베딩 / Dense·Sparse·하이브리드 / RRF / 리랭킹 |
| 30~60분 | **실습 1** | [`part1/`](part1/) — 청킹 → 임베딩 → 원리 검증 → Vector Search 2.0 검색·필터·리랭킹 |
| 60~75분 | 이론 2 | kNN vs ANN(ScaNN), Live API 양방향 스트리밍, 에이전트 구조 |
| 75~80분 | 배포 시작 | `gcloud run deploy` 실행 후 **기다리지 않고** 실습 2로 이동 |
| 80~105분 | **실습 2** | [`part2/`](part2/) — 상품 컬렉션 질의: 텍스트 / 이미지 / RRF 가중치 / 리랭킹 → 앱 코드 대조 |
| 105~115분 | 라이브 데모 | QR 생성 → 스마트폰 카메라·음성으로 에이전트 사용 |
| 115~120분 | 마무리 | 자원 정리, Q&A |

> Cloud Run 소스 빌드에 약 5분이 걸립니다. **배포를 실습 2보다 먼저 걸어두는 것**이 이 일정의 핵심입니다.

---

## 폴더 구조

```
smx-multimodal-agent/
├── part1/                      # 실습 1 — 멀티모달 검색 원리 (노트북 + README)
├── part2/                      # 실습 2 — VS2 검색 엔진 + 에이전트 배포 (노트북 + app/ + README)
├── install.sh                  # 실습 환경 프로비저닝 스크립트
├── session2_index_builder.py   # Part 2 컬렉션·인덱스 빌더 (install.sh가 호출)
└── README.md                   # 이 문서
```

---

# 실습 준비 (약 10분)

## 1. Gemini API Key 확인

#### 1-1. 메뉴에서 `credential` 을 검색해 진입합니다.
![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/1.png)

#### 1-2. 생성돼 있는 `GeminiLabKey` 의 `Show key` 를 눌러 값을 복사해 둡니다.
![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/2.png)

> [!NOTE]
> 이 키는 **Part 2의 Cloud Run 배포에서만** 사용합니다 (Gemini Live API 호출용).
> Part 1 노트북은 프로젝트 ADC 자격 증명을 그대로 쓰므로 키 입력 단계가 없습니다.

<br>

## 2. Workbench 실행 및 실습자료 다운로드

#### 2-1. 상단 검색 메뉴에서 `workbench` 를 입력해 'Workbench' 메뉴를 클릭합니다.
![image](https://raw.githubusercontent.com/jk1333/handson/main/images/6/1.png)

#### 2-2. `Open Jupyterlab` 버튼을 눌러 환경에 접속합니다.
![image](https://raw.githubusercontent.com/cheeunlim/agent-engine-lab/main/images/workbench_open.png)

#### 2-3. Terminal에 진입해 실습자료를 내려받습니다.

```bash
git clone https://github.com/cheeunlim/smx-multimodal-agent
```

<br>

## 3. 환경 프로비저닝

```bash
cd ~/smx-multimodal-agent
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
| 5 | **인덱스 빌더를 백그라운드로 구동** (`session2_index_builder.py`) | 20~40분 |

> [!IMPORTANT]
> 5번은 **백그라운드에서 계속 돌아갑니다.** 스크립트가 끝나도 인덱싱은 진행 중입니다.
> 이 시간이 이론 1 + 실습 1 구간에 흡수되므로, 그대로 두고 **Part 1을 시작하면 됩니다.**

진행 상황은 아래로 볼 수 있습니다.

```bash
tail -f index_builder.log
```

## 실습 준비 완료!

<br>

---

# 실습

## [실습 1 — 멀티모달 검색 엔진의 원리](part1/) (30분)

비디오를 10초 단위로 쪼개고, Gemini Embedding 2로 텍스트·이미지·비디오를 **하나의 벡터 공간**에 담습니다.
직접 짠 코사인 유사도·BM25 하이브리드로 원리를 확인한 뒤, 같은 질의를 Vector Search 2.0에 던져
**손으로 구현한 `alpha` 결합이 VS2에서는 RRF `weights` 한 줄로 대체된다**는 점을 눈으로 확인합니다.

## [실습 2 — VS2 검색 엔진과 실시간 쇼핑 에이전트](part2/) (25분 + 데모 10분)

약 10만 건의 Amazon 상품 컬렉션에 텍스트·이미지 질의를 던지고, RRF 가중치를 뒤집어 순위가 어떻게
흔들리는지 확인합니다. 마지막에 **방금 실행한 호출이 배포 중인 에이전트의 엔진과 같은 코드**임을
소스로 대조하고, QR 코드로 스마트폰에서 직접 사용해 봅니다.

> **순서 주의**: `part2/README.md` 의 **Cloud Run 배포 명령을 먼저 실행**한 뒤 노트북으로 이동하세요.

<br>

---

## 참고: 백그라운드 인덱싱 상태 점검

```bash
gcloud vector-search operations list --location=asia-northeast1 \
  --format='value(name.basename(), done, error.message)'
```

`install.sh` 는 총 **4개**의 장기 작업(LRO)을 만듭니다.

| # | 작업 | Part 2 시작 전 완료 필요? |
| :--- | :--- | :--- |
| 1 | 컬렉션 생성 (`.../collections/amazon-product-768-compact`) | ✅ **필수** |
| 2 | 데이터 임포트 (`ImportDataObjectsMetadata`) | ✅ **필수** |
| 3 | 텍스트 인덱스 생성 (`.../indexes/idx-text-embedding`) | 아니오 |
| 4 | 이미지 인덱스 생성 (`.../indexes/idx-image-embedding`) | 아니오 |

**1번과 2번만 `done: true` 면 실습 2를 바로 진행할 수 있습니다** (약 20분).
인덱스(3·4번)가 아직 없어도 검색은 **kNN 완전탐색**으로 정상 동작하며, 조금 느릴 뿐입니다.
노트북 10번 단계에서 인덱스 유무를 직접 출력해 확인합니다.

개별 작업을 자세히 보려면:

```bash
gcloud vector-search operations describe <OPERATION_NAME> --location=asia-northeast1
```

### 인덱싱이 멈춘 것 같다면

```bash
tail -30 ~/smx-multimodal-agent/index_builder.log
```

빌더가 중간에 끝났다면 컬렉션과 데이터는 이미 올라가 있으므로,
**`install.sh` 를 다시 실행하지 말고** 강사에게 알려 주세요
(재실행하면 데이터가 중복 임포트됩니다).
