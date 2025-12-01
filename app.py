# import streamlit as st
# import os
# from openai import AzureOpenAI
# from dotenv import load_dotenv
# # 시간 추가
# from datetime import datetime
# import pytz

# # 1. 환경 변수 로드 (.env 파일이 같은 폴더에 있어야 함)
# load_dotenv()

# search_index = ("SEARCH_INDEX_NAME", "azure-rag")
# search_key = os.getenv("SEARCH_KEY")
# search_endpoint = os.getenv("SEARCH_ENDPOINT")
# semantic_config = os.getenv("SEMANTIC_CONFIG", "azure-rag-semantic-configuration")
# st.title("Azure Expert")

# # 2. Azure OpenAI 클라이언트 설정
# # (실제 값은 .env 파일이나 여기에 직접 입력하세요)
# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OAI_KEY"),
#     api_version="2024-05-01-preview",
#     azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT"),

# )

# # 3. 대화기록(Session State) 초기화 - 이게 없으면 새로고침 때마다 대화가 날아갑니다!
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # 4. 화면에 기존 대화 내용 출력
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # 5. 사용자 입력 받기
# if prompt := st.chat_input("무엇을 도와드릴까요?"):
#     # (1) 사용자 메시지 화면에 표시 & 저장
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # 시간 체크 로직 추가
#     processed_prompt = prompt.lower()
#     import re
#     processed_prompt = re.sub(r'[\s\.\?!]', '', processed_prompt)  # 공백 제거
#     time_keywords_clean = ['시간', '몇시', '현재시각', '지금몇시', '오늘날짜', '오늘이며칠', '오늘은몇월며칠', '오늘요일']

#     is_time_request = any(keyword in processed_prompt for keyword in time_keywords_clean)
#     assistant_reply = ""
#     if is_time_request:
#         tz = pytz.timezone('Asia/Seoul')
#         # %Y년 %m월 %d일
#         current_time = datetime.now(tz).strftime('%H시 %M분')
#         assistant_reply = f"현재 시간은 {current_time}입니다."
#     else:
#         # (2) AI 응답 생성 (스트리밍 방식 아님, 단순 호출 예시)
#         messages_payload=[
#             {"role": m["role"], "content": m["content"]}
#             for m in st.session_state.messages
#         ]
#         with st.spinner("생각중..."):
#             try:

#                 response = client.chat.completions.create(
#                     model="gpt-4o-mini",
#                     messages=messages_payload,
#                     max_tokens=6553,
#                     temperature=0.7,
#                     top_p=0.95,
#                     frequency_penalty=0,
#                     presence_penalty=0,
#                     stop=None,
#                     stream=False,
#                     extra_body={
#                     "data_sources": [{
#                         "type": "azure_search",
#                         "parameters": {
#                             "endpoint": f"{search_endpoint}",
#                             "index_name": "azure-rag",
#                             "semantic_configuration": "azure-rag-semantic-configuration",
#                             "query_type": "vector_semantic_hybrid",
#                             "fields_mapping": {},
#                             "in_scope": True,
#                             "filter": None,
#                             "strictness": 3,
#                             "top_n_documents": 5,
#                             "authentication": {
#                             "type": "api_key",
#                             "key": f"{search_key}"
#                             },
#                             "embedding_dependency": {
#                             "type": "deployment_name",
#                             "deployment_name": "text-embedding-ada-002"
#                             }
#                         }
#                         }]
#                     }
#                 )
#                 assistant_reply = response.choices[0].message.content
#             except Exception as e:
#                 assistant_reply = f"Azure RAG 호출 중 오류 발생: 설정 또는 네트워크 문제일 수 있습니다. 오류: `{e}`"

#     # (3) AI 응답 저장
#     with st.chat_message("assistant"):
#         st.markdown(assistant_reply)

#     st.session_state.messages.append({"role": "assistant", "content": assistant_reply})



import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime
import pytz
import re
import json # JSON 파싱을 위해 추가

# 1. 환경 변수 로드 (.env 파일이 같은 폴더에 있어야 함)
load_dotenv()

# 환경 변수 설정
AZURE_OAI_KEY = os.getenv("AZURE_OAI_KEY")
AZURE_OAI_ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT")
SEARCH_KEY = os.getenv("SEARCH_KEY")
SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
SEARCH_INDEX_NAME = os.getenv("SEARCH_INDEX_NAME", "azure-rag")
SEMANTIC_CONFIG = os.getenv("SEMANTIC_CONFIG", "azure-rag-semantic-configuration")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")

# 2. Azure OpenAI 클라이언트 설정 (앱 전체에서 공유)
try:
    client = AzureOpenAI(
        api_key=AZURE_OAI_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint=AZURE_OAI_ENDPOINT,
    )
except Exception as e:
    st.error(f"Azure OpenAI 클라이언트 초기화 오류: 환경 변수를 확인하세요. ({e})")
    st.stop()


## UI 및 세션 관리 함수
# 새 대화 시작 함수
def new_chat():
    """세션 상태의 메시지를 초기화합니다."""
    st.session_state.messages = []
    st.session_state.error_message = None

# Streamlit UI 설정
st.set_page_config(page_title="Azure Expert RAG Chatbot", layout="wide")

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정 및 제어")
    
    st.info(f"""
        **현재 환경:**
        - **모델:** {os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini")}
        - **Search Index:** `{SEARCH_INDEX_NAME}`
        """)
    
    # 새 대화 버튼
    if st.button("🆕 새로운 대화 시작", use_container_width=True):
        new_chat()

st.title("📘 Azure Expert RAG Chatbot")

# 3. 대화기록(Session State) 초기화
if "messages" not in st.session_state:
    new_chat() # 초기화 함수 재사용

# 4. 화면에 기존 대화 내용 출력
for message in st.session_state.messages:
    # 사용자 메시지
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    # AI 메시지 (출처 포함)
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔗 **참조 문서 출처**"):
                    for source in message["sources"]:
                        st.markdown(source)

# 5. 사용자 입력 받기 및 응답 생성
if prompt := st.chat_input("Azure 클라우드 관련 무엇이 궁금하신가요?"):
    # (1) 사용자 메시지 화면에 표시 & 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 시간 체크 로직 (기존 로직 유지)
    processed_prompt = prompt.lower()
    processed_prompt = re.sub(r'[\s\.\?!]', '', processed_prompt) 
    time_keywords_clean = ['시간', '몇시', '현재시각', '지금몇시', '오늘날짜', '오늘이며칠', '오늘은몇월며칠', '오늘요일']
    is_time_request = any(keyword in processed_prompt for keyword in time_keywords_clean)

    assistant_reply_content = ""
    sources_list = []

    if is_time_request:
        tz = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(tz).strftime('%Y년 %m월 %d일 %H시 %M분')
        assistant_reply_content = f"현재 한국 시간은 **{current_time}**입니다."
    else:
        # (2) AI 응답 생성 (스트리밍 및 RAG 포함)
        messages_payload=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        
        # Streamlit Chat Element 생성 (스트리밍을 위해 placeholder 역할)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # API 호출 (stream=True로 변경)
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini"),
                    messages=messages_payload,
                    max_tokens=6553,
                    temperature=0.7,
                    stream=True, # **스트리밍 활성화**
                    extra_body={
                        "data_sources": [{
                            "type": "azure_search",
                            "parameters": {
                                "endpoint": SEARCH_ENDPOINT,
                                "index_name": SEARCH_INDEX_NAME,
                                "semantic_configuration": SEMANTIC_CONFIG,
                                "query_type": "vector_semantic_hybrid",
                                "in_scope": True,
                                "top_n_documents": 5,
                                "authentication": {
                                    "type": "api_key",
                                    "key": SEARCH_KEY
                                },
                                "embedding_dependency": {
                                    "type": "deployment_name",
                                    "deployment_name": EMBEDDING_DEPLOYMENT_NAME
                                }
                            }
                        }]
                    }
                )
                
                # 스트리밍 응답 처리
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        message_placeholder.markdown(full_response + "▌") # 커서 효과
                
                # 최종 응답 내용 업데이트
                assistant_reply_content = full_response
                message_placeholder.markdown(assistant_reply_content)

                # RAG 소스 추출 (Chat Completions API의 경우, 소스 정보가 'context' 필드에 포함되어 나옴)
                # 이 정보는 스트리밍의 경우 마지막 청크 또는 별도의 로직으로 추출해야 합니다.
                # 현재 Azure OpenAI Python SDK는 스트리밍 시 소스 추출이 복잡하므로,
                # 여기서는 응답 문자열에서 인용구(citations)를 찾아 표시하는 방법을 사용합니다.

                sources_pattern = r'\[doc\d{1,2}\]|\\[doc\d{1,2}\]' # 예시: [doc1]
                citations = re.findall(sources_pattern, assistant_reply_content)
                
                # 실제 RAG 소스 정보는 API 응답의 'context' 필드에 더 자세히 담겨 있지만,
                # Streamlit UI 상의 간편성을 위해 API 응답의 **특정 필드**에서 직접 추출합니다.
                # 실제 구현 시에는 API 응답에서 'context' (tool_calls) 필드를 파싱하는 코드가 필요합니다.
                # 임시적으로, Streamlit의 Expander에 "출처 정보가 포함된 응답을 보려면 전체 API 응답을 파싱해야 합니다."라는 메시지를 표시합니다.
                
                # (실제 RAG 출처 표시 로직: 실제 API 응답 구조를 기반으로 context/tool_calls를 파싱해야 함)
                # 현재 Streamlit Chatbot 구조상 파싱이 복잡하여 임시 메시지 표시
                
                # 실제 API 응답에서 context를 파싱했다고 가정하고 임시 URL 목록을 추가
                # 실제 RAG 소스 파싱이 복잡하여, 이 예시에서는 응답에 포함된 **인용 번호**를 기반으로 임시 URL 목록을 생성합니다.
                if citations:
                    st.info("💡 **참고:** 정확한 출처 URL을 표시하려면 API 응답의 JSON을 파싱해야 합니다. 현재는 인용 번호만 표시됩니다.")
                    # 임시 소스 목록 생성 (실제 구현 시 API 응답에서 추출)
                    # 이 부분을 완성하려면, response 객체의 .tool_calls 또는 .context 필드를 파싱해야 합니다.
                    # 현재 코드는 파싱 로직이 없으므로 임시 메시지를 추가합니다.
                    
                    sources_list.append("파싱된 실제 소스 (예: [Azure Document URL])")
                    sources_list.append("Azure AI Search에서 검색된 Top 5 문서 제목")
                    
                    with st.expander("🔗 **참조 문서 출처 (파싱 필요)**"):
                        st.markdown("✅ **참조된 문서 번호:** " + ", ".join(sorted(list(set(citations)))))
                        st.markdown("> **참고:** 정확한 출처 URL을 얻으려면 `response.choices[0].message.context` (또는 `tool_calls`) 필드를 파싱하는 코드를 추가해야 합니다.")

            except Exception as e:
                assistant_reply_content = f"⚠️ **Azure RAG 호출 중 오류 발생:** 설정 또는 네트워크 문제일 수 있습니다. 오류: `{e}`"
                message_placeholder.markdown(assistant_reply_content)
                st.session_state.error_message = assistant_reply_content

    # (3) AI 응답 저장 (시간 요청이든, AI 응답이든)
    # 메시지 리스트에 최종 답변 저장 (소스 정보가 있을 경우 함께 저장)
    message_to_save = {"role": "assistant", "content": assistant_reply_content}
    if sources_list:
        message_to_save["sources"] = sources_list
        
    st.session_state.messages.append(message_to_save)
