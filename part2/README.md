# Part 2 — Vector Search 2.0 검색 엔진 + 멀티모달 쇼핑 에이전트 배포

Gemini Live API와 크로스모달 검색 엔진을 결합한 실시간 대화형 쇼핑 에이전트 **LensMosaic** 을 배포하고,
그 안에서 돌아가는 Vector Search 2.0 질의 엔진을 노트북에서 직접 실행해 봅니다.

본 예제는 [LensMosaic](https://github.com/kazunori279/lens-mosaic/tree/main) 을 기반으로 하며,
상품 데이터는 [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) 를 사용합니다.

## 폴더 구조

```
part2/
├── vector_search_agent.ipynb   # 실습 2 노트북 (약 25분)
├── app/                        # 에이전트 코드
│   ├── main.py                 # Gemini Live 에이전트 + FastAPI 서버
│   ├── prompt.py               # 에이전트 프롬프트
│   ├── common.py               # 설정값 (컬렉션, 모델, 리전)
│   ├── embedding_vector.py     # Gemini Embedding 2 + Vector Search 2.0 질의 엔진
│   ├── session.py              # 세션 상태 관리
│   └── static/                 # 프론트엔드
├── qr.py                       # QR 코드 생성기
├── download_agent_card.py      # A2A Agent Card 다운로드
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 진행 순서 (약 40분)

| 단계 | 소요 | 내용 |
| :--- | :--- | :--- |
| **1** | 약 1분 | **Cloud Run 배포 명령 실행** — 실습 1 완료 후 실행 (백그라운드 빌드 진행 중 **이론 2** 로 이동) |
| **2** | 약 25분 | `vector_search_agent.ipynb` 실습 (시작 시점엔 이미 배포 빌드가 완료되어 있습니다) |
| **3** | 약 10분 | QR 코드 생성 → 스마트폰에서 라이브 데모 |

> **진행 순서 안내**: 1단계(배포)를 2단계(노트북 실습)보다 먼저 진행해 주세요.
> Cloud Run 소스 빌드에 약 5분이 소요되므로, 배포 명령을 먼저 실행해 두면 빌드가 진행되는 동안 원활하게 **이론 2 세션**을 진행할 수 있습니다.
> 터미널에서 배포 명령을 실행해 둔 상태로 강의 세션에 참여하시면 됩니다.

---

## 1단계. Cloud Run 배포 시작 (백그라운드 빌드)

아래 명령어를 실행해 Cloud Run 배포를 시작합니다. `(Y/n)` 선택이 나오면 엔터를 입력합니다.

```bash
cd ~/multimodal-agent/part2
gcloud run deploy lens-mosaic \
  --source . \
  --region "asia-southeast1" \
  --concurrency 500 --cpu 2 --memory 4Gi --timeout 3600 \
  --min-instances 1 --max-instances 1 --execution-environment=gen2
```

배포 명령을 실행한 후 빌드 완료를 대기할 필요 없이 다음 단계(이론 2)로 진행합니다. 빌드는 백그라운드에서 계속 진행되며, 이어지는 이론 세션 동안 완료됩니다.

> **참고**: 실습 환경의 조직 정책(Domain Restricted Sharing)으로 인해 CLI에서는 공개 접근(`allUsers`) 설정이 제한됩니다. 서비스 자체는 정상 배포되며, **3단계에서 QR 코드를 생성하기 전에 GCP 콘솔에서 공개 접근을 허용**합니다.

---

## 2단계. 실습 노트북 — `vector_search_agent.ipynb`

JupyterLab에서 `part2/vector_search_agent.ipynb` 를 열고 셀을 위에서부터 순서대로 실행합니다.
(`Ctrl + Enter` 또는 메뉴의 `Run > Run Selected Cell`)

실습 준비 단계에서 실행한 `install.sh` 가 `amazon-product-768-compact` 컬렉션과 약 10만 건(99,426건)의 상품 데이터를
백그라운드로 올려 두었으므로, **이 노트북은 컬렉션이나 인덱스를 만들지 않습니다.**
프로비저닝 대기 시간이 0이며, 첫 셀부터 바로 검색을 실행합니다.
(ScaNN 인덱스 생성이 진행 중이더라도 kNN 완전탐색 방식으로 검색이 정상 동작하므로 실습 진행에는 지장이 없습니다.)

| # | 내용 |
| :--- | :--- |
| 1 | 클라이언트 초기화 · 계정/프로젝트 확인 및 컬렉션 핸들 |
| 2 | 백그라운드 인덱싱 완료 및 에러 로그(`index_builder.log`) 확인 |
| 3 | 컬렉션 스키마 확인 — dense 벡터 필드 2개의 의미 |
| 4 | 상품 카탈로그 프리뷰 + 공용 헬퍼 정의 |
| 5 | 텍스트 질의 ➔ `text_embedding` 검색 |
| 6 | 이미지 질의 ➔ `image_embedding` 검색 (크로스모달) |
| 7 | **[핵심] RRF 가중치 실험** — `[1.35, 0.65]` ↔ `[0.65, 1.35]` |
| 8 | Ranking API 리랭킹 |
| 9 | 에이전트 프롬프트와 `find_items` 툴 호출 흐름 |
| 10 | **나노바나나로 테스트용 상품 이미지 생성하기** (선택) 🎨 |

### 노트북 실행 전 확인

컬렉션 준비 상태와 빌더 로그는 **노트북 2번 단계 셀에서 바로 출력**되며, 터미널에서 미리 볼 수도 있습니다.

```bash
gcloud vector-search operations list --location=asia-southeast1
tail -n 20 ~/multimodal-agent/index_builder.log
```

**컬렉션 생성**과 **데이터 임포트** 두 작업이 `done: True` 이면 노트북 전체를 실행할 수 있습니다.
인덱스 생성 작업 2개는 진행 중이더라도 검색은 정상 동작합니다(인덱스 없이도 kNN 완전탐색 방식으로 처리됩니다).
출력 해석 및 409 에러/계정 트러블슈팅 방법은 저장소 루트 [`README.md`](../README.md#자주-발생하는-에러--트러블슈팅-가이드) 를 참고하세요.

> 노트북이 사용하는 파이썬 패키지(`google-cloud-vectorsearch`, `google-genai`,
> `google-cloud-discoveryengine`, `Pillow`)는 모두 `install.sh` 가 설치해 둡니다.

---

## 3단계. 공개 접근 허용 · QR 코드 생성 및 모바일 라이브 데모

스마트폰에서 에이전트에 접속하려면 **QR 코드를 만들기 전에 먼저 Cloud Run 콘솔에서 공개 접근을 허용**해야 합니다.

#### 1. 콘솔에서 공개 접근 허용하기 및 URL 복사

실습 환경의 조직 정책으로 인해 CLI 명령으로는 공개 접근이 허용되지 않으므로, 콘솔에서 직접 설정합니다.

1. GCP 콘솔 상단 메뉴에서 `cloud run` 을 검색해 진입한 뒤 `lens-mosaic` 서비스를 클릭합니다.
2. `Security` 탭으로 이동합니다.
3. `Authentication` 항목에서 **`Allow public access`** 를 선택합니다.
4. 하단의 **`Save`** 를 클릭합니다.

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/3.png)

<br>

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/4.png)

<br>

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/5.png)

<br>

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/6.png)

<br>

5. 설정을 저장한 뒤, 콘솔 화면 상단 **URL** 옆의 복사 버튼을 눌러 서비스 주소를 복사합니다. (또는 터미널에서 아래 명령으로 URL을 확인할 수 있습니다.)

```bash
gcloud run services describe lens-mosaic --region asia-southeast1 --format="value(status.url)"
```

#### 2. QR 코드 생성

`-------CLOUD RUN URL-------` 을 복사한 주소로 교체 후 실행합니다.

```bash
cd ~/multimodal-agent/part2
python qr.py -------CLOUD RUN URL------- -o my_qrcode.png
```

생성된 `my_qrcode.png` 파일을 열고, 스마트폰 카메라로 인식해 에이전트를 실행합니다.

#### 3. 테스트용 이미지 준비하기 — 나노바나나(Nano Banana) 활용 🎨

현장에서 카메라로 비출 실물 소품이 다양하지 않다면, **나노바나나**(`gemini-2.5-flash-image`)로 원하는 상품을 생성하거나 직접 그림을 그려 노트북 화면에 크게 띄워 놓고 스마트폰 카메라로 비춰 보세요!

*   **방법 A** (노트북에서 바로 생성 — 가장 빠름):
    *   `vector_search_agent.ipynb` 맨 마지막 **10번 셀**(`TEST_IMAGE_PROMPT`)에 원하는 상품을 입력하고 실행하면 화면에 이미지가 바로 출력됩니다.
*   **방법 B** ([Google AI Studio](https://aistudio.google.com/) / 나노바나나에서 생성하거나 직접 그리기):
    1.  [Google AI Studio](https://aistudio.google.com/) 에 접속해 모델로 **Nano Banana** (`gemini-2.5-flash-image`)를 선택합니다.
    2.  아래 **ABO 카탈로그 맞춤 프롬프트**를 입력해 이미지를 생성하거나, 간단한 스케치/그림을 그린 뒤 실사 상품 이미지로 변환해 달라고 요청합니다.
    3.  생성된 이미지를 클릭해 화면에 크게 띄웁니다.

> [!TIP]
> **카탈로그 맞춤 프롬프트 예시** (본 상품 카탈로그에는 가방·신발·가전·주방·조명·가구·주얼리가 풍부하며, 원피스/셔츠 같은 일반 의류는 포함되어 있지 않습니다):
> *   가방: `빈티지 브라운 가죽 백팩 제품 사진` / `A vintage brown leather backpack on a wooden table`
> *   헤드폰: `매트 블랙 무선 노이즈캔슬링 헤드폰` / `Matte black wireless over-ear headphones`
> *   주방/텀블러: `민트색 스테인리스 보온 텀블러` / `A pastel mint stainless steel coffee tumbler`
> *   조명/인테리어: `따뜻한 조명의 황동 데스크 스탠드 램프` / `A modern brass desk lamp with a warm bulb`
> *   신발: `주황색 트레일 러닝화 운동화` / `Bright orange trail running shoes`

#### 4. 음성으로 사용해 보기

에이전트 우측 하단의 마이크 버튼을 눌러 음성 입력을 활성화한 뒤,
모바일 카메라로 실물 소품(또는 위에서 나노바나나로 만든 화면 속 이미지)을 비추면서 말해 봅니다.

```
이 가방이랑 어울리는 소품을 추천해줘
이거랑 비슷한 디자인의 다른 제품 보여줘
```

- 화면을 비추는 것만으로 **외형이 유사한 상품이 자동으로 검색**됩니다
  (노트북 6번 단계의 `image_embedding` 검색과 같은 경로입니다).
- 음성으로 요구사항을 말하면 에이전트가 `find_items` 툴을 호출해 추천 상품을 렌더링합니다
  (노트북 5번·8번 단계의 텍스트 검색 + Ranking API 리랭킹 경로입니다).

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/10.png)

---

## (보너스) Agent Registry 등록 및 검색 테스트

추가 실습을 원하시는 분들을 위한 선택 과정입니다. 배포한 에이전트를 A2A(Agent-to-Agent) 프로토콜용
Agent Card로 내보내고, Agent Registry에 등록하여 검색 동작을 확인합니다.

#### 1. Agent Card 생성

```bash
cd ~/multimodal-agent/part2
python download_agent_card.py -------CLOUD RUN URL-------
```

#### 2. Agent Registry에 등록

```bash
gcloud alpha agent-registry services create lens-mosaic \
  --location=global \
  --display-name="LensMosaic" \
  --agent-spec-type=a2a-agent-card \
  --agent-spec-content=agent-card.json
```

#### 3. 등록된 에이전트 검색

```bash
gcloud alpha agent-registry agents search --location=global --search-string="쇼핑"
```

---

## 실습 완료!

`install.sh` 로 생성한 Vector Search 컬렉션과 GCS 버킷은 Qwiklabs 실습 세션 종료 시 자동으로 정리됩니다.
개인 GCP 프로젝트에서 실습하신 경우 아래 명령어로 직접 리소스를 정리할 수 있습니다.

```bash
gcloud run services delete lens-mosaic --region asia-southeast1
gcloud storage rm -r gs://$(gcloud config get-value project)-vs2
```
