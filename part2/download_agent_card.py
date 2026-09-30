import json
import sys
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def _build_fallback_agent_card(target_url: str) -> dict:
    """서버 측 a2a-sdk 버전 차이로 500 오류가 발생할 때 로컬에서 동일한 스키마의 Agent Card를 생성합니다."""
    instruction = "An ADK Agent"
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from app.prompt import AGENT_PROMPT
        instruction = AGENT_PROMPT
    except Exception:
        prompt_path = Path(__file__).parent / "app" / "prompt.py"
        if prompt_path.exists():
            raw = prompt_path.read_text(encoding="utf-8")
            m = re.search(r'"""\\?\n?(.*?)"""', raw, re.DOTALL)
            if m:
                instruction = m.group(1)

    find_items_desc = (
        "Find shopping items that match one or more product description queries.\n\n"
        "**Invocation Condition:** \n"
        "1. 사용자가 '유사한 아이템을 찾아달라'고 할 때 카메라 분석 후 즉시 호출합니다.\n"
        "2. 사용자가 '어울리는 물건'이나 '선물' 등을 추천해 달라고 할 때, `google_search` 도구를 먼저 호출하여 "
        "트렌드 검색을 완료한 직후에 보강된 키워드로 이어서 호출합니다.\n\n"
        "**Tool Description:**\n\n"
        "Use this tool when you want to show the user product candidates on screen.\n"
        "Provide a list of descriptive English product-search queries. The tool\n"
        "searches and publishes the matched items to the UI, then yields the top item\n"
        "names back to the live agent. ranking_query is used for the final Ranking API\n"
        "rerank across all merged candidates."
    )

    return {
        "additionalInterfaces": [],
        "capabilities": {
            "extensions": [],
            "pushNotifications": False,
            "stateTransitionHistory": False,
            "streaming": True,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "description": "An ADK Agent",
        "documentationUrl": "",
        "iconUrl": "",
        "name": "mm_agent",
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "provider": {
            "organization": "Google Cloud",
            "url": "https://cloud.google.com",
        },
        "security": [],
        "securitySchemes": {},
        "signatures": [],
        "skills": [
            {
                "description": instruction,
                "examples": [],
                "id": "mm_agent",
                "inputModes": [],
                "name": "model",
                "outputModes": [],
                "security": [],
                "tags": ["llm"],
            },
            {
                "description": "google_search",
                "examples": [],
                "id": "mm_agent-google_search",
                "inputModes": [],
                "name": "google_search",
                "outputModes": [],
                "security": [],
                "tags": ["llm", "tools"],
            },
            {
                "description": find_items_desc,
                "examples": [],
                "id": "mm_agent-find_items",
                "inputModes": [],
                "name": "find_items",
                "outputModes": [],
                "security": [],
                "tags": ["llm", "tools"],
            },
        ],
        "supportsAuthenticatedExtendedCard": False,
        "url": target_url,
        "version": "0.0.1",
    }


def download_and_modify_agent_card(target_url):
    # URL 끝의 슬래시(/) 처리 후 최종 요청 URL 생성
    base_url = target_url if target_url.endswith('/') else f"{target_url}/"
    request_url = f"{base_url}.well-known/agent-card.json"
    
    print(f"요청 중: {request_url}")
    
    try:
        # 브라우저처럼 보이도록 User-Agent 추가 (일부 서버의 차단 방지)
        req = Request(request_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        # 1. URL 요청 및 데이터 다운로드
        with urlopen(req) as response:
            raw_data = response.read().decode('utf-8')
            json_data = json.loads(raw_data)
        
        # 2. root의 'url' 키값을 입력받은 target_url로 교체
        json_data['url'] = target_url
    except HTTPError as e:
        print(f"⚠️ 서버 응답 오류 ({e.code} {e.reason}) — 로컬 에이전트 정의에서 직접 agent-card.json 을 생성합니다.")
        json_data = _build_fallback_agent_card(target_url)
    except URLError as e:
        print(f"URL 오류 발생: {e.reason}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("오류: 다운로드한 데이터가 올바른 JSON 형식이 아닙니다.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 'agent-card.json' 파일로 저장 (들여쓰기 2칸으로 깔끔하게 포맷팅)
    output_filename = "agent-card.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
        
    print(f"성공: '{output_filename}' 파일이 생성되었고 url 값이 '{target_url}'로 교체되었습니다.")

if __name__ == "__main__":
    # 인자값(URL) 확인
    if len(sys.argv) < 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"사용법: python {script_name} <URL>")
        print(f"예시: python {script_name} https://example.com")
        sys.exit(1)
        
    user_url = sys.argv[1]
    download_and_modify_agent_card(user_url)