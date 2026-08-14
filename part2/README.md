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
| **1** | 약 1분 | **Cloud Run 배포 명령 실행** — 실습 1이 끝나면 바로. 빌드 완료를 기다리지 않고 **이론 2** 로 이동 |
| **2** | 약 25분 | `vector_search_agent.ipynb` 실습 (시작 시점엔 이미 배포 빌드가 끝나 있습니다) |
| **3** | 약 10분 | QR 코드 생성 → 스마트폰에서 라이브 데모 |

> **중요**: 1단계와 2단계의 순서를 바꾸지 마세요.
> Cloud Run 소스 빌드에 약 5분이 걸리는데, 배포를 먼저 걸어두면 그 시간이 **이론 2 시간에 그대로 흡수**됩니다.
> 터미널 한 개에서 배포를 실행해 두고, **같은 터미널을 닫지 않은 채** 강의로 돌아가면 됩니다.

---

## 1단계. Cloud Run 배포 시작 (기다리지 마세요)

`-----GEMINI_API_KEY-----` 부분을 실습 준비 과정에서 복사해 둔 키로 교체한 뒤 실행합니다.
`(Y/n)` 선택이 나오면 엔터를 입력합니다.

```bash
cd ~/smx-multimodal-agent/part2
gcloud run deploy lens-mosaic \
  --source . \
  --region "asia-northeast1" \
  --set-env-vars GEMINI_API_KEY="-----GEMINI_API_KEY-----" \
  --allow-unauthenticated \
  --concurrency 500 --cpu 2 --memory 4Gi --timeout 3600 \
  --min-instances 1 --max-instances 1 --execution-environment=gen2
```

`--allow-unauthenticated` 플래그가 배포와 동시에 공개 접근을 허용합니다.

명령을 실행했으면 **완료를 기다리지 마세요.** 빌드는 백그라운드에서 계속 진행되며, 이어지는 이론 2 동안 완료됩니다.

> **참고**: 배포 명령이 `allUsers` 바인딩 실패로 끝나면 조직 정책(Domain Restricted Sharing) 때문입니다.
> 이 경우 아래 [폴백 — 콘솔에서 공개 접근 허용하기](#폴백--콘솔에서-공개-접근-허용하기) 절차를 따르세요.
> 서비스 자체는 정상적으로 배포되며, 공개 접근 설정만 수동으로 해주면 됩니다.

---

## 2단계. 실습 노트북 — `vector_search_agent.ipynb`

JupyterLab에서 `part2/vector_search_agent.ipynb` 를 열고 셀을 위에서부터 순서대로 실행합니다.
(`Ctrl + Enter` 또는 메뉴의 `Run > Run Selected Cell`)

실습 준비 단계에서 실행한 `install.sh` 가 `amazon-product-768-compact` 컬렉션과 약 10만 건의 상품 데이터를
백그라운드로 올려 두었으므로, **이 노트북은 컬렉션이나 인덱스를 만들지 않습니다.**
프로비저닝 대기 시간이 0이며, 첫 셀부터 바로 검색을 실행합니다.
(ScaNN 인덱스 2개도 같은 스크립트가 만들지만, **아직 완성되지 않았어도 실습에는 지장이 없습니다.**
인덱스가 없으면 kNN 완전탐색으로 처리되며 조금 느릴 뿐입니다 — 10번 단계에서 직접 확인합니다.)

| # | 내용 |
| :--- | :--- |
| 1 | 클라이언트 초기화 및 컬렉션 핸들 |
| 2 | 백그라운드 인덱싱 완료 확인 |
| 3 | 컬렉션 스키마 확인 — dense 벡터 필드 2개의 의미 |
| 4 | 상품 카탈로그 프리뷰 + 공용 헬퍼 정의 |
| 5 | 텍스트 질의 ➔ `text_embedding` 검색 |
| 6 | 이미지 질의 ➔ `image_embedding` 검색 (크로스모달) |
| 7 | **[핵심] RRF 가중치 실험** — `[1.35, 0.65]` ↔ `[0.65, 1.35]` |
| 8 | Ranking API 리랭킹 |
| 9 | 메타데이터 필터 결합 |
| 10 | ANN(ScaNN) vs kNN — 코드는 그대로, 속도만 다르다 |
| 11 | `app/embedding_vector.py` 소스 대조 |
| 12 | 에이전트 프롬프트와 `find_items` 툴 호출 흐름 |

### 노트북 실행 전 확인

컬렉션 준비 상태는 노트북 2번 단계에서 확인하지만, 터미널에서 미리 볼 수도 있습니다.

```bash
gcloud vector-search operations list --location=asia-northeast1
```

**컬렉션 생성**과 **데이터 임포트** 두 작업이 `done: true` 이면 노트북 전체를 실행할 수 있습니다.
인덱스 생성 작업 2개는 아직 진행 중이어도 검색은 동작합니다(인덱스 없이 kNN 완전탐색으로 처리되며, 조금 느릴 뿐입니다).
출력 해석 방법은 저장소 루트 `README.md` 의 "백그라운드 작업 상태 점검" 절을 참고하세요.

> 노트북이 사용하는 파이썬 패키지(`google-cloud-vectorsearch`, `google-genai`,
> `google-cloud-discoveryengine`, `Pillow`)는 모두 `install.sh` 가 설치해 둡니다.

---

## 3단계. QR 코드 생성 및 모바일 라이브 데모

#### 1. 배포 완료 확인 및 URL 복사

1단계 터미널로 돌아가 배포가 끝났는지 확인하고, 출력된 Service URL을 복사합니다.
콘솔에서 확인하려면 메뉴에서 `cloud run` 검색 → `lens-mosaic` 클릭 → 상단 URL 옆 복사 버튼을 누릅니다.

```bash
gcloud run services describe lens-mosaic --region asia-northeast1 --format="value(status.url)"
```

#### 2. QR 코드 생성

`-------CLOUD RUN URL-------` 을 복사한 주소로 교체 후 실행합니다.

```bash
cd ~/smx-multimodal-agent/part2
python qr.py -------CLOUD RUN URL------- -o my_qrcode.png
```

생성된 `my_qrcode.png` 파일을 열고, 스마트폰 카메라로 인식해 에이전트를 실행합니다.

#### 3. 테스트용 이미지 만들기 (선택)

주변에 마땅한 상품이 없다면 이미지를 생성해서 화면에 띄워놓고 카메라로 비춰도 됩니다.

1. 클라우드 콘솔에서 `studio` 를 검색해 Agent Platform Studio 로 진입합니다.
2. 좌측 상단 `+` 버튼 → `Image` 를 클릭합니다.
3. 프롬프트 입력 후 생성합니다.

```
흰색 꽃무늬 원피스를 입은 여성 마네킨
```

4. 생성된 이미지를 클릭해 크게 띄웁니다.

#### 4. 음성으로 사용해 보기

에이전트 우측 하단의 마이크 버튼을 눌러 음성 입력을 활성화한 뒤,
모바일 카메라로 상품(또는 위에서 만든 이미지)을 비추면서 말해 봅니다.

```
어울리는 가방을 추천해줘
```

- 화면을 비추는 것만으로 **외형이 유사한 상품이 자동으로 검색**됩니다
  (노트북 6번 셀의 `image_embedding` 검색과 같은 경로입니다).
- 음성으로 요구사항을 말하면 에이전트가 `find_items` 툴을 호출해 추천 상품을 렌더링합니다
  (노트북 5번·8번 셀의 텍스트 검색 + Ranking API 리랭킹 경로입니다).

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/10.png)

---

## 폴백 — 콘솔에서 공개 접근 허용하기

> 이 절차는 **`--allow-unauthenticated` 가 조직 정책(Domain Restricted Sharing, DRS)에 막혀
> 실패했을 때만** 수행합니다. 배포가 정상적으로 끝났다면 건너뛰세요.

DRS 정책이 켜져 있는 프로젝트에서는 `allUsers` 에 대한 IAM 바인딩이 거부될 수 있습니다.
이때는 서비스만 배포된 상태이므로, 콘솔에서 공개 접근을 직접 켜 줍니다.

1. 콘솔 메뉴에서 `cloud run` 을 검색해 진입한 뒤 `lens-mosaic` 서비스를 클릭합니다.
2. `Security` 탭으로 이동합니다.
3. `Authentication` → `Allow public access` 를 선택합니다.
4. `Save` 를 클릭합니다.

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/3.png)

<br>

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/4.png)

<br>

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/5.png)

<br>

![image](https://raw.githubusercontent.com/jk1333/handson/main/images/7/6.png)

<br>

콘솔 대신 CLI로 처리하려면 아래 명령도 사용할 수 있습니다(동일하게 DRS 영향을 받습니다).

```bash
gcloud run services add-iam-policy-binding lens-mosaic \
  --region asia-northeast1 \
  --member="allUsers" \
  --role="roles/run.invoker"
```

---

## (보너스) Agent Registry 등록 및 검색 테스트

시간이 남는 참가자를 위한 선택 과제입니다. 배포한 에이전트를 A2A(Agent-to-Agent) 프로토콜용
Agent Card로 내보내고, Agent Registry에 등록해 검색되는지 확인합니다.

#### 1. Agent Card 생성

```bash
cd ~/smx-multimodal-agent/part2
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

`install.sh` 로 만든 Vector Search 컬렉션과 GCS 버킷은 Qwiklab 세션이 끝나면 함께 정리됩니다.
개인 프로젝트에서 실습했다면 아래로 직접 정리하세요.

```bash
gcloud run services delete lens-mosaic --region asia-northeast1
gcloud storage rm -r gs://$(gcloud config get-value project)-vs2
```
