import streamlit as st
import pandas as pd
from utils.langgraph_rag import run_rag
from utils.index_qa import query_similar_complaints
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

st.set_page_config(layout="wide")

st.title("📞 가스안전공사 민원 자동응답 생성 및 유사민원 검색")
st.text(
    """AI가 규정 및 법률을 기반으로 답변 초안을 생성해주고 해당 민원과 유사한 과거 민원을 검색합니다."""
)
st.session_state.references = []
st.session_state.similar_complaints = []
QA_PATH = "data/qa_csv/merged_qa.csv"
CHROMA_QA_PATH = "data/vector_qa_db"
QA_MENT = "우리공사 홈페이지 질의응답 게시판을 이용하여 주신 고객님께 진심으로 감사 드리며 고객님의 질의에 대하여 아래와 같이 답변 드립니다.\n"
edit_col, rag_col = st.columns([6, 4])

with edit_col:
    # CSV 파일에서 민원 목록 불러오기
    try:
        complaints = pd.read_csv(QA_PATH)
        # 열 이름 매핑 - '제목'과 '질문'열을 사용
        complaints = complaints.rename(columns={"질문": "내용"})
    except Exception as e:
        st.error(f"민원목록을 불러오는 중 오류가 발생했습니다: {e}")

    # 민원 선택
    st.markdown("#### 민원 답변 작성 ####")
    selected_title = st.selectbox(
        "처리할 민원을 선택하세요", complaints["제목"], key="complaint_selector"
    )
    selected_complaint = complaints[complaints["제목"] == selected_title].iloc[0]
    with st.container():
        st.markdown(
            f"""
        <div style="background-color:#F0F2F6; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            {selected_complaint["내용"]}
        </div>
        """,
            unsafe_allow_html=True,
        )
    if "response" not in st.session_state:
        st.session_state.response = ""
    if "references" not in st.session_state:
        st.session_state.references = []
    if "similar_complaints" not in st.session_state:
        st.session_state.similar_complaints = []

    # 버튼을 누를 때 session_state 값을 업데이트하는 함수
    def generate_response():
        # 스트리밍 모드 활성화
        st.session_state.streaming = True
        # 스트리밍 시작 전에 references 초기화
        st.session_state.references = []

    # 버튼 생성
    if st.button(
        "🧠 AI 답변 초안 생성 하기",
        on_click=generate_response,
        use_container_width=True,
    ):
        stream_generator, context_info = run_rag(
            selected_complaint["내용"], streaming=True
        )
        st.session_state.references = context_info["retrieved_documents"]

        full_response = ""

        markdown_placeholder = st.empty()
        for chunk in stream_generator:
            full_response += chunk
            full_response = full_response.replace("\n", "  \n")
            markdown_placeholder.markdown(full_response, unsafe_allow_html=True)


# 민원이 선택될 때마다 유사 민원 검색
def search_similar_complaints():
    # 선택된 민원의 제목과 내용을 합친 쿼리 생성
    query_text = f"{selected_complaint['제목']} {selected_complaint['내용']}"

    # Chroma DB 로드
    try:
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory=CHROMA_QA_PATH,
            embedding_function=embeddings,
            collection_name="complaints_qa",
        )

        # 유사 민원 검색
        similar_complaints = query_similar_complaints(vectordb, query_text)
        st.session_state.similar_complaints = similar_complaints
    except Exception as e:
        st.error(f"유사 민원 검색 중 오류가 발생했습니다: {e}")
        st.session_state.similar_complaints = []


# 민원 선택 시 유사 민원 검색 실행
if st.session_state.get("complaint_selector") is not None:
    search_similar_complaints()

with rag_col:
    st.markdown("#### 참고 자료 ####")

    rag, search = st.tabs(["답변 참고 문서", "유사 민원"])

    with rag:
        st.caption("AI가 답변에 참고한 문서 내용입니다.")
        if hasattr(st.session_state, "references") and st.session_state.references:
            for doc in st.session_state.references:
                with st.expander(doc.metadata.get("filename", "문서")):
                    st.write(doc.page_content)
        else:
            st.info("답변 생성 버튼을 클릭하면 참고 문서가 표시됩니다.")

    with search:
        st.caption("현재 민원과 유사한 과거 민원입니다.")
        if st.session_state.get("similar_complaints"):
            for complaint in st.session_state.similar_complaints:
                metadata = complaint.get("metadata", {})
                with st.expander(metadata.get("title", "제목 없음")):
                    st.write(f"**민원내용:** {metadata.get('question', '내용 없음')}")

                    answer_text = metadata.get("answer", "답변 없음")

                    # <질의요지>와 <답변내용> 문자열이 있는 경우 줄바꿈 처리
                    answer_text = QA_MENT + answer_text
                    answer_text = answer_text.replace(
                        "<질의요지>", "  \n  \n<질의요지>  \n"
                    )
                    answer_text = answer_text.replace(
                        "<답변내용>", "  \n  \n<답변내용>  \n"
                    )
                    answer_text = answer_text.replace("ㅇ", "  \nㅇ")

                    answer_text = answer_text.split("<답변자 정보>")
                    answer_text = answer_text[0]
                    # 마크다운 형식으로 표시하여 줄바꿈이 제대로 적용되도록 함
                    st.markdown(f"**답변:**\n\n{answer_text}")
        else:
            st.info("유사한 민원을 찾는 중입니다...")
