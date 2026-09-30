# **[Part 1] 멀티모달 임베딩 & Vector Search 2.0 검색** (약 30분)

**Gemini Embedding 2**로 비디오·이미지·텍스트를 하나의 3072차원 공간에 올리고, 그 위에서 **Vertex AI Vector Search 2.0**으로 크로스모달 하이브리드 검색을 수행하는 실습입니다.

*   **실습 노트북**: [multimodal_search.ipynb](multimodal_search.ipynb)
*   **실행 환경**: GCP Workbench (JupyterLab)
*   **소요 시간**: 약 25분 (코드 셀 14개 + 정리용 Raw 셀 1개)
*   **인증**: 프로젝트 ADC (`genai.Client(vertexai=True, ...)`)

---

## **설계 원칙**

> **로컬 연산은 원리를 보여주는 glass box, 최종 결과는 Vector Search 2.0이 반환한다.**

NumPy로 직접 구현한 코사인 유사도(`registry_matrix @ query_vector`)는 검색 엔진 내부에서 실제로 수행되는 연산 과정을 이해하기 위한 교육용 구현입니다.
동일한 질의를 관리형 서비스(Vector Search 2.0)로 실행하여 **결과와 지연시간을 나란히 비교**하는 것이 이 노트북의 핵심(11번 코드 셀)입니다.

| 개념 | 로컬 직접 구현 | Vector Search 2.0 |
| :--- | :--- | :--- |
| 유사도 연산 | NumPy 코사인 완전탐색 (`@`) | `semantic_search` (kNN) |
| 멀티모달 공간 | 비디오 청크 + 텍스트 앵커 투영 | 3072차원 단일 통합 벡터 |
| 하이브리드 검색 | — | `semantic_search` + `text_search` (`ReciprocalRankFusion`) |
| 정밀도 리랭킹 | — | Vertex AI Ranking API |

---

## **셀 구성** (14 코드 셀 + 1 Raw 셀)

| # | 구분 | 내용 | 예상 소요 |
| :--- | :--- | :--- | :--- |
| 1 | 설정 | 패키지 설치 + 자동 커널 재시작 가드 | ~40초 (1회) |
| 2 | 설정 | Project ID 및 활성 계정 자동 감지 · **ADC 기반 Vertex GenAI 클라이언트** 생성 | ~5초 |
| 3 | **VS2** | 0단계: 컬렉션 생성 (409 중복 생성 자동 방어 포함) | ~10초 |
| 4 | 전처리 | 1단계: FFmpeg 스트림 복사로 10초 단위 비디오 청킹 | ~15초 |
| 5 | 임베딩 | 2단계: `generate_multimodal_embedding()` 정의 | 즉시 |
| 6 | 임베딩 | 청크 10개 **병렬** 임베딩 (`ThreadPoolExecutor`) | ~5초 |
| 7 | 분석 | 코사인 유사도 — 최유사 / 최이질 세그먼트 + **인라인 영상 3개 나란히 재생** | ~6초 |
| 8 | 분석 | **t-SNE 멀티모달 궤적 시각화** — 비디오 청크 궤적 + 텍스트 앵커 동시 투영 + 클립 드롭다운 재생 | ~4초 |
| 9 | 검색 | Dense 단독 시맨틱 검색 (크로스모달 체감) — 상위 3개 구간 인라인 재생 | ~6초 |
| 10 | 데이터 | 3단계: 레지스트리 로드(135MB) + **약 1,000건 서브샘플링** + 인메모리 행렬 구축 | ~10초 |
| 11 | **VS2** | **병렬 배치 업서트** (250건 × 4배치, 409 방어 포함) | ~20초 |
| 12 | **VS2** | ⭐ **동일 질의 비교** (① 로컬 NumPy 완전탐색 vs ② VS2 `semantic_search` kNN) + 지연시간 | ~5초 |
| 13 | **VS2** | `semantic_search` + `text_search`를 `batch_search` 내장 **RRF** (`weights`)로 융합 | ~5초 |
| 14 | 최적화 | 4단계: Vertex AI Ranking API 리랭킹 | ~5초 |
| 15 | 정리 | 5단계: 컬렉션 삭제 (**Raw 셀** — 실수 방지를 위해 실행되지 않음) | ~1분 |

---

## **핵심 최적화 3가지**

1.  **사전 연산 레지스트리 재사용**
    이미지 4,606건 + 비디오 청크 199건의 3072차원 임베딩을 미리 계산해 `.pkl`로 배포합니다.
    실습에서는 다운로드만 하므로, 원래 수십 분 걸릴 임베딩 구간이 10번 셀 몇 초로 줄어듭니다.
2.  **임베딩·업서트 병렬화**
    동시성 상수는 **두 개**입니다.

    | 상수 | 위치 | 대상 | 값 |
    | :--- | :--- | :--- | :--- |
    | `MAX_WORKERS` | 6번 셀 | 비디오 청크 및 텍스트 앵커 임베딩 생성 | 8 |
    | `UPSERT_WORKERS` | 11번 셀 | `BatchCreateDataObjects` 배치 전송 | 8 |

    API 할당량 초과(429 Resource Exhausted)가 발생할 경우 값을 4로 조정할 수 있습니다.
3.  **ANN 인덱스 미생성** (kNN 완전탐색 사용)
    Vector Search 2.0은 **인덱스가 없어도** 시맨틱 검색·전문검색·RRF 하이브리드를 모두 수행합니다.
    인덱스 생성은 수십 분이 걸리므로 Part 1에서는 만들지 않고, Part 2에서 ScaNN 인덱스가 걸린 컬렉션을 사용합니다.

---

## **사용 방법**

1.  **노트북 실행**: JupyterLab 왼쪽 패널에서 `part1` > `multimodal_search.ipynb`를 더블 클릭합니다.
2.  **순서대로 실행**: 1번 셀에서 커널이 1회 자동 재시작됩니다. 재시작 후 **처음 셀부터** 다시 순서대로 실행하세요.
3.  **인증 및 계정 확인**: 2번 셀은 Workbench의 ADC를 사용하며, 현재 연결된 `Project ID`와 `Active Account`를 출력합니다.
4.  **질의 및 앵커 바꿔 보기**: `ANCHOR_TEXTS`, `DENSE_QUERY`, `COMPARE_QUERY`, `RRF_QUERY`, `RERANK_QUERY` 상수를 바꾸면 바로 실험해 볼 수 있습니다.

---

## **데이터 및 스키마**

*   **원본 영상**: `gs://ai-multimodal-data/team_usa_tech.mp4`
*   **사전 연산 레지스트리**: `https://storage.googleapis.com/ai-multimodal-data/full_dataset_registry.pkl`
    (135MB · 총 4,805항목 = 이미지 4,606 + 비디오 청크 199 · 3072차원)
*   **업서트 대상**: 비디오 청크 199개 전량 + 이미지 800개 = 999건 (250건 × 4배치)

```python
# 컬렉션 스키마
data_schema.properties = {
    "description": {"type": "string"},   # ➔ text_search(전문검색) 대상
    "media_type":  {"type": "number"},   # ➔ 1 = image, 0 = video_chunk
    "source":      {"type": "string"},   # ➔ 원본 출처 식별자
}

vector_schema = {
    "content_embedding": DenseVectorField(dimensions=3072,
        vertex_embedding_config=VertexEmbeddingConfig(model_id="gemini-embedding-2")),
}
```

---

## **주요 기술 스택**

*   **Google GenAI SDK** (`google-genai`) — `gemini-embedding-2`
*   **Vertex AI Vector Search 2.0** (`google-cloud-vectorsearch`) — 서버리스 컬렉션, kNN 검색, 내장 RRF (`semantic_search` + `text_search`)
*   **Vertex AI Ranking API** (`google-cloud-discoveryengine`) — 매니지드 크로스 인코더 리랭커
*   **FFmpeg Muxer** — 무손실 스트림 복사 세그먼팅

---

> [!WARNING]
> 전체 실습이 모두 완료된 후에는 **마지막 Raw 셀의 타입을 `Code`로 변경하여 실행**함으로써 서버리스 컬렉션을 정리해 주세요.
> Part 2 실습은 별도의 상품 컬렉션(`amazon-product-768-compact`)을 사용하므로, 이 자원 정리 셀은 전체 실습의 **맨 마지막**에 실행하시면 됩니다.
