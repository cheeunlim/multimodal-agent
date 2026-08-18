import time
from datetime import datetime
import google.auth
from google.cloud import vectorsearch_v1
from google.api_core import retry as google_retry
from google.api_core import exceptions

# ---------------------------------------------------------
# 1. 인증 및 환경 설정
# ---------------------------------------------------------
_, PROJECT_ID = google.auth.default()
LOCATION = "asia-northeast1"
COLLECTION_ID = "amazon-product-768-compact"

vector_search_service_client = vectorsearch_v1.VectorSearchServiceClient()

# SDK 수준 기본 Retry (단기 트랜지언트 네트워크 재시도용)
custom_retry = google_retry.Retry(
    predicate=google_retry.if_exception_type(
        exceptions.ServiceUnavailable,     # 503
        exceptions.DeadlineExceeded,       # 504
        exceptions.InternalServerError,    # 500
        exceptions.BadGateway              # 502
    ),
    initial=2.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=300.0  # 누적 재시도 시간을 5분으로 확장
)

def execute_with_step_retry(step_name, func, max_retries=5, initial_delay=30):
    """
    500 에러 등 임시 서버 오류 발생 시, 
    해당 Step 전체를 백오프(Exponential Backoff) 방식으로 보장 재시도하는 함수
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [{step_name}] 시도 {attempt}/{max_retries} 시작...")
            return func()
        except exceptions.AlreadyExists:
            print(f"⚠️ [{step_name}] 리소스가 이미 존재합니다. 다음 단계로 진행합니다.")
            return None
        except exceptions.GoogleAPICallError as e:
            print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] [{step_name}] Google API 에러 발생 (Code: {e.code}): {e.message}")
            if attempt == max_retries:
                print(f"❌ [{step_name}] 최대 재시도 횟수({max_retries}회)를 초과하여 최종 실패했습니다.")
                raise e
            
            print(f"⏱️ {delay}초 대기 후 [{step_name}] 단계를 재시도합니다...")
            time.sleep(delay)
            delay *= 2  # 지수 백오프 (30초 -> 60초 -> 120초...)
        except Exception as e:
            print(f"❌ [{step_name}] 예상치 못한 예외 발생: {e}")
            raise e

def wait_for_lro_clean(operation, timeout_seconds=1200, poll_interval=15):
    """
    SDK 내부 'neither response nor error set' 예외 방지용 안전 폴링
    """
    start_time = time.time()
    while True:
        # 최신 Operation pb 직접 가져오기
        operation._operation = operation._refresh(retry=custom_retry)
        pb_op = operation.operation

        if pb_op.done:
            if pb_op.HasField("error"):
                raise exceptions.from_grpc_status(
                    status_code=pb_op.error.code,
                    message=pb_op.error.message,
                    errors=(pb_op.error,),
                    response=pb_op,
                )
            return pb_op

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f"LRO 작업이 지정된 시간({timeout_seconds}초) 내에 완료되지 않았습니다.")

        time.sleep(poll_interval)


# ---------------------------------------------------------
# 2. Step 1: Collection 생성 (500 에러 발생 시 Step 전체 재시도)
# ---------------------------------------------------------
def step1_create_collection():
    data_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
    }

    vector_schema = {
        "image_embedding": {"dense_vector": {"dimensions": 768}},
        "text_embedding": {"dense_vector": {"dimensions": 768}}
    }

    collection = vectorsearch_v1.Collection(
        data_schema=data_schema,
        vector_schema=vector_schema,
    )

    create_collection_req = vectorsearch_v1.CreateCollectionRequest(
        parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
        collection_id=COLLECTION_ID,
        collection=collection,
    )

    # 1) API 요청 (500 에러 발생 지점)
    operation = vector_search_service_client.create_collection(
        request=create_collection_req,
        retry=custom_retry,
        timeout=120.0
    )
    # 2) LRO 안전 대기
    wait_for_lro_clean(operation, timeout_seconds=600)
    print(f"✅ Collection created at {datetime.now()}")

# Step 1 실행 (실패 시 30초, 60초, 120초... 쉬면서 최대 5번 재시도)
execute_with_step_retry("Step 1: Collection 생성", step1_create_collection)
time.sleep(10)


# ---------------------------------------------------------
# 3. Step 2: Data Import
# ---------------------------------------------------------
def step2_import_data():
    import_req = vectorsearch_v1.ImportDataObjectsRequest(
        name=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION_ID}",
        gcs_import={
            "contents_uri": f"gs://{PROJECT_ID}-vs2/data/",
            "error_uri": f"gs://{PROJECT_ID}-vs2/error/",
        },
    )

    operation = vector_search_service_client.import_data_objects(
        request=import_req,
        retry=custom_retry,
        timeout=120.0
    )
    wait_for_lro_clean(operation, timeout_seconds=1200, poll_interval=15)
    print(f"✅ Import data finished at {datetime.now()}")

execute_with_step_retry("Step 2: Data Import", step2_import_data)
time.sleep(10)


# ---------------------------------------------------------
# 4. Step 3 & 4: Index 생성
# ---------------------------------------------------------
# 한 컬렉션에 인덱스 생성 요청은 한 번에 하나만 큐잉된다. 앞 인덱스가 만들어지는
# 중에 다음 인덱스를 요청하면 409 Aborted (unable to queue the operation) 가 난다.
# 빌드가 얼마나 걸릴지 미리 알 수 없으므로(99,426건 실측 약 68분) 대기 타임아웃에
# 기대지 않는다. 큐가 열릴 때까지 일정 간격으로 요청을 다시 던지기만 하면 된다.
INDEX_QUEUE_DEADLINE = 10800  # 큐가 열릴 때까지 최대 3시간
INDEX_QUEUE_POLL = 120        # 재요청 간격 (초)

def request_index(index_field: str):
    def _action():
        index_id = f"idx-{index_field.replace('_', '-')}"
        index = vectorsearch_v1.Index(
            index_field=index_field,
            store_fields=["name", "description"],
        )

        create_index_req = vectorsearch_v1.CreateIndexRequest(
            parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{COLLECTION_ID}",
            index_id=index_id,
            index=index,
        )

        started = time.time()
        while True:
            try:
                op = vector_search_service_client.create_index(
                    request=create_index_req,
                    retry=custom_retry,
                    timeout=120.0
                )
                print(f"📨 Index ({index_field}) 생성 요청됨 at {datetime.now()}")
                return op
            except exceptions.Conflict as e:
                # 409 는 두 가지다. AlreadyExists 는 바깥에서 "이미 존재"로 처리하므로 넘긴다.
                if isinstance(e, exceptions.AlreadyExists):
                    raise
                # 나머지 409(Aborted, unable to queue)는 앞 인덱스가 아직 만들어지는
                # 중이라는 뜻이다. 에러가 아니라 차례를 기다리는 상태다.
                waited = int(time.time() - started)
                if waited > INDEX_QUEUE_DEADLINE:
                    raise
                print(f"⏳ [{datetime.now().strftime('%H:%M:%S')}] Index ({index_field}) 큐 대기 중 "
                      f"— 앞 인덱스 빌드가 끝나면 자동으로 요청됩니다 ({waited // 60}분 경과)")
                time.sleep(INDEX_QUEUE_POLL)

    return execute_with_step_retry(f"Index 생성 요청 [{index_field}]", _action)

# 인덱스가 아직 없어도 검색은 kNN 완전탐색으로 동작한다. 따라서 무엇이 실패해도
# 워크숍을 멈추지 않고, 무엇이 남았는지 알려주기만 한다.
# 요청을 먼저 다 걸어 둔 뒤에 대기한다 — image_embedding 요청이 text_embedding
# 빌드 완료를 기다리는 동안 큐를 계속 두드리므로, 별도의 순서 제어가 필요 없다.
pending = []
unfinished = []

for field in ("text_embedding", "image_embedding"):
    try:
        op = request_index(field)
    except Exception as e:
        print(f"⚠️ Index ({field}) 생성 요청 실패: {e}")
        unfinished.append(field)
        continue
    if op is not None:  # None 이면 이미 존재하는 인덱스다
        pending.append((field, op))

for field, op in pending:
    try:
        wait_for_lro_clean(op, timeout_seconds=INDEX_QUEUE_DEADLINE, poll_interval=60)
        print(f"✅ Index ({field}) created at {datetime.now()}")
    except Exception as e:
        print(f"⚠️ Index ({field}) 대기 중 문제 발생: {e}")
        print(f"   서버측 작업은 계속 진행 중일 수 있습니다. 아래로 확인하세요:")
        print(f"   gcloud vector-search operations list --location={LOCATION}")
        unfinished.append(field)

if unfinished:
    print(f"\n⚠️ 파이프라인 종료 ({datetime.now()}) — 아직 준비되지 않은 인덱스: {', '.join(unfinished)}")
    print("   인덱스가 없어도 검색은 kNN 완전탐색으로 정상 동작합니다. 실습을 그대로 진행하세요.")
else:
    print(f"\n🎉 모든 파이프라인 작업이 성공적으로 완료되었습니다! ({datetime.now()})")