import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
# 시간 추가
from datetime import datetime
import pytz

# 1. 환경 변수 로드 (.env 파일이 같은 폴더에 있어야 함)
load_dotenv()

search_index = ("SEARCH_INDEX_NAME", "azure-rag")
search_key = os.getenv("SEARCH_KEY")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
semantic_config = os.getenv("SEMANTIC_CONFIG", "azure-rag-semantic-configuration")
st.title("Azure Expert")

# 2. Azure OpenAI 클라이언트 설정
# (실제 값은 .env 파일이나 여기에 직접 입력하세요)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"),
    api_version="2024-05-01-preview",
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT"),

)

# 3. 대화기록(Session State) 초기화 - 이게 없으면 새로고침 때마다 대화가 날아갑니다!
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 화면에 기존 대화 내용 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 입력 받기
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # (1) 사용자 메시지 화면에 표시 & 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 시간 체크 로직 추가
    processed_prompt = prompt.lower()
    import re
    processed_prompt = re.sub(r'[\s\.\?!]', '', processed_prompt)  # 공백 제거
    time_keywords_clean = ['시간', '몇시', '현재시각', '지금몇시', '오늘날짜', '오늘이며칠', '오늘은몇월며칠', '오늘요일']

    is_time_request = any(keyword in processed_prompt for keyword in time_keywords_clean)
    assistant_reply = ""
    if is_time_request:
        tz = pytz.timezone('Asia/Seoul')
        # %Y년 %m월 %d일
        current_time = datetime.now(tz).strftime('%H시 %M분')
        assistant_reply = f"현재 시간은 {current_time}입니다."
    else:
        # (2) AI 응답 생성 (스트리밍 방식 아님, 단순 호출 예시)
        messages_payload=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        with st.spinner("생각중..."):
            try:

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_payload,
                    max_tokens=6553,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False,
                    extra_body={
                    "data_sources": [{
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": f"{search_endpoint}",
                            "index_name": "azure-rag",
                            "semantic_configuration": "azure-rag-semantic-configuration",
                            "query_type": "vector_semantic_hybrid",
                            "fields_mapping": {},
                            "in_scope": True,
                            "filter": None,
                            "strictness": 3,
                            "top_n_documents": 5,
                            "authentication": {
                            "type": "api_key",
                            "key": f"{search_key}"
                            },
                            "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": "text-embedding-ada-002"
                            }
                        }
                        }]
                    }
                )
                assistant_reply = response.choices[0].message.content
            except Exception as e:
                assistant_reply = f"Azure RAG 호출 중 오류 발생: 설정 또는 네트워크 문제일 수 있습니다. 오류: `{e}`"

    # (3) AI 응답 저장
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
