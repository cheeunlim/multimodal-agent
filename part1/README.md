# **[Part 1] 멀티모달 임베딩 & Vector Search 2.0 검색 (약 30분)**

**Gemini Embedding 2**로 비디오·이미지·텍스트를 하나의 3072차원 공간에 올리고, 그 위에서 **Vertex AI Vector Search 2.0**으로 크로스모달 하이브리드 검색을 수행하는 실습입니다.

*   **실습 노트북**: [multimodal_search.ipynb](multimodal_search.ipynb)
*   **실행 환경**: GCP Workbench (JupyterLab)
*   **소요 시간**: 약 30분 (코드 셀 18개 + 정리용 Raw 셀 1개)
*   **인증**: 프로젝트 ADC (`genai.Client(vertexai=True, ...)`) — **API 키 입력 단계가 없습니다**

---

## **설계 원칙**

> **로컬 연산은 원리를 보여주는 glass box, 최종 결과는 Vector Search 2.0이 반환한다.**

NumPy로 직접 짠 코사인 유사도와 `SimpleBM25`는 "검색 엔진 내부에서 실제로 일어나는 연산"을 열어 보이기 위한 교육 장치입니다.
같은 질의를 매니지드 서비스로 다시 실행해 **결과와 지연시간을 나란히 대조**하는 것이 이 노트북의 핵심(15번 셀)입니다.

| 개념 | 직접 짜 보는 것 | Vector Search 2.0 |
| :--- | :--- | :--- |
| 유사도 연산 | NumPy 코사인 완전탐색 | `semantic_search` (kNN) |
| 키워드 검색 | `SimpleBM25` | `text_search(data_field_names=["description"])` |
| 가중치 결합 | `alpha * dense + (1-alpha) * sparse` | `ReciprocalRankFusion(weights=[...])` |
| 개인화 | 설명문에 태그 문자열 결합 | `filter={"tag": {"$eq": "Me"}}` |

---

## **셀 구성 (18 코드 셀 + 1 Raw 셀)**

| # | 구분 | 내용 | 예상 소요 |
| :--- | :--- | :--- | :--- |
| 1 | 설정 | 패키지 설치 + 자동 커널 재시작 가드 | ~40초 (1회) |
| 2 | 설정 | Project ID 자동 감지 · **ADC 기반 Vertex GenAI 클라이언트** 생성 | ~5초 |
| 3 | **VS2** | 컬렉션 생성 (LRO 완료까지 대기 · 이미 있으면 재사용) | ~1분 |
| 4 | 전처리 | FFmpeg 스트림 복사로 10초 단위 청킹 | ~15초 |
| 5 | 임베딩 | `generate_multimodal_embedding()` 정의 | 즉시 |
| 6 | 임베딩 | 청크 10개 **병렬** 임베딩 (`ThreadPoolExecutor`) | ~20초 |
| 7 | 분석 | 코사인 유사도 — 최유사 / 최이질 세그먼트 + 프레임 비교 | ~3초 |
| 8 | 분석 | **t-SNE 시간축 궤적 시각화** | ~3초 |
| 9 | 검색 | Dense 단독 시맨틱 검색 (크로스모달 체감) | ~3초 |
| 10 | 캡션 | Gemini Flash 청크 설명문 **병렬** 생성 | ~25초 |
| 11 | 검색 | `SimpleBM25` 구현 + 토크나이저 | 즉시 |
| 12 | 검색 | `alpha` 가중치 하이브리드 (0.8 vs 0.3 대조) | ~5초 |
| 13 | 데이터 | 레지스트리 로드(135MB) + **약 1,000건 서브샘플링** + `"Me"` 태그를 데이터 필드로 부여 | ~30초 |
| 14 | **VS2** | **병렬 배치 업서트** (250건 × 4배치) | ~20초 |
| 15 | **VS2** | ⭐ **동일 질의 3-way 비교** (로컬 완전탐색 / 로컬 하이브리드 / VS2 kNN) + 지연시간 | ~5초 |
| 16 | **VS2** | `semantic_search` + `text_search`를 `batch_search` 내장 **RRF**로 융합 | ~5초 |
| 17 | **VS2** | `filter={"tag": {"$eq": "Me"}}` 개인화 필터 | ~5초 |
| 18 | 최적화 | Ranking API 리랭킹 + 비디오 크라우딩 필터 | ~8초 |
| 19 | 정리 | 컬렉션 삭제 (**Raw 셀** — 실수 방지를 위해 실행되지 않음) | ~1분 |

---

## **핵심 최적화 3가지**

1.  **사전 연산 레지스트리 재사용**
    이미지 4,606건 + 비디오 청크 199건의 3072차원 임베딩을 미리 계산해 `.pkl`로 배포합니다.
    실습에서는 다운로드만 하므로, 원래 수십 분 걸릴 임베딩 구간이 13번 셀 30초로 줄어듭니다.
2.  **임베딩·캡션·업서트 병렬화**
    순차 호출 시 약 7분 걸리던 구간이 약 45초로 줄어듭니다. 동시성 상수는 **두 개**입니다.

    | 상수 | 위치 | 대상 | 값 |
    | :--- | :--- | :--- | :--- |
    | `MAX_WORKERS` | 6번 셀 | 임베딩 생성 · 캡션 생성 (10번 셀에서도 재사용) | 8 |
    | `UPSERT_WORKERS` | 14번 셀 | `BatchCreateDataObjects` 배치 전송 | 8 |

    둘 다 드라이런 실측 최적값입니다. 429(Resource Exhausted)가 보이면 4로 낮추고, 16으로 올리면 스로틀링으로 오히려 느려집니다.
3.  **ANN 인덱스 미생성 (kNN 완전탐색 사용)**
    Vector Search 2.0은 **인덱스가 없어도** 시맨틱 검색·전문검색·RRF 하이브리드·메타데이터 필터를 모두 수행합니다.
    1만 건 기준 인덱스 생성은 약 30분이 걸리므로 Part 1에서는 만들지 않고, Part 2에서 ANN(ScaNN) 인덱스가 걸린 컬렉션과 대조합니다.

---

## **사용 방법**

1.  **노트북 실행**: JupyterLab 왼쪽 패널에서 `part1` > `multimodal_search.ipynb`를 더블 클릭합니다.
2.  **순서대로 실행**: 1번 셀에서 커널이 1회 자동 재시작됩니다. 재시작 후 **처음 셀부터** 다시 순서대로 실행하세요.
3.  **키 입력 없음**: 2번 셀은 Workbench의 서비스 계정 자격 증명(ADC)을 그대로 사용합니다. `GeminiLabKey` 는 Part 2의 Cloud Run 배포에서만 필요합니다.
4.  **질의 바꿔 보기**: `DENSE_QUERY`, `HYBRID_QUERY`, `COMPARE_QUERY`, `RRF_QUERY`, `FILTER_QUERY`, `RERANK_QUERY` 상수를 바꾸면 검색어가 바뀝니다.

---

## **데이터 및 스키마**

*   **원본 영상**: `gs://ai-multimodal-data/team_usa_tech.mp4`
*   **사전 연산 레지스트리**: `https://storage.googleapis.com/ai-multimodal-data/full_dataset_registry.pkl`
    (135MB · 총 4,805항목 = 이미지 4,606 + 비디오 청크 199 · 3072차원)
*   **업서트 대상**: 비디오 청크 199개 전량 + 이미지 800개 = 약 1,000건 (250건 × 4배치)

```python
# 컬렉션 스키마
data_schema.properties = {
    "description": {"type": "string"},   # ➔ text_search 대상
    "tag":         {"type": "string"},   # ➔ filter 대상 ("Me" / "Public")
    "media_type":  {"type": "number"},   # ➔ 1 = image, 0 = video_chunk
    "source":      {"type": "string"},   # ➔ 크라우딩 필터 기준
}

vector_schema = {
    "content_embedding": DenseVectorField(dimensions=3072,
        vertex_embedding_config=VertexEmbeddingConfig(model_id="gemini-embedding-2")),
}
```

> 값을 채우지도 않던 `description_embedding`(SparseVectorField)은 제거했습니다.
> 희소 검색은 `description` **데이터 필드**를 `text_search`로 직접 검색하는 방식으로 대체됩니다.
> 마찬가지로 업서트에서 값을 넣지 않던 `start` / `end` 도 스키마에서 뺐습니다 (선언만 되고 쓰이지 않는 필드는 혼란만 줍니다).

---

## **주요 기술 스택**

*   **Google GenAI SDK** (`google-genai`) — `gemini-embedding-2` / `gemini-3.5-flash`
*   **Vertex AI Vector Search 2.0** (`google-cloud-vectorsearch`) — 서버리스 컬렉션, kNN 검색, 내장 RRF
*   **Vertex AI Ranking API** (`google-cloud-discoveryengine`) — 매니지드 리랭커
*   **FFmpeg Muxer** — 무손실 스트림 복사 세그먼팅
*   **SimpleBM25 / NumPy** — 검색 원리 설명용 인메모리 구현

---

> [!WARNING]
> 워크숍이 모두 끝난 뒤에는 **19번 Raw 셀의 타입을 `Code`로 바꿔 실행**하여 서버리스 컬렉션을 삭제해 주세요.
> Part 2는 별도의 상품 컬렉션(`amazon-product-768-compact`)을 사용하므로, 이 셀은 **맨 마지막**에 실행하면 됩니다.
