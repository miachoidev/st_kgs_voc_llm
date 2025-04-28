import streamlit as st
import pandas as pd
import os

st.title("📁 RAG 문서 관리")

st.info("AI가 민원 응답에 참고할 문서를 관리합니다.")


# 색인된 문서 정보 로드
@st.cache_data
def load_indexed_documents():
    # 색인된 문서 정보가 저장된 CSV 파일이 있는지 확인
    if os.path.exists("data/indexed_documents.csv"):
        return pd.read_csv("data/indexed_documents.csv")
    else:
        # 테스트용 예시 데이터 (실제 색인 전)
        return pd.DataFrame(
            {
                "category": ["유권해석", "법령", "지침"],
                "filename": [
                    "민원처리지침.pdf",
                    "가스안전법_2023.hwp",
                    "위험물관리규정.docx",
                ],
            }
        )


# 파일 업로드 (현재 기능은 없고 UI만 표시)
# with st.expander("새 문서 업로드 (색인은 별도 프로세스로 진행)"):
#     st.file_uploader("파일 업로드 하기", type=["pdf", "hwp", "docx", "txt"])
#     st.caption("※ 업로드된 파일의 실제 색인은 관리자가 별도로 진행합니다.")

# 색인된 문서 목록 표시
st.subheader("색인된 문서 목록")
docs_df = load_indexed_documents()

# 카테고리별 필터링 옵션
if not docs_df.empty:
    categories = ["전체"] + sorted(docs_df["category"].unique().tolist())
    selected_category = st.selectbox("카테고리 필터", categories)

    if selected_category != "전체":
        filtered_df = docs_df[docs_df["category"] == selected_category]
    else:
        filtered_df = docs_df

    # 데이터프레임 표시
    st.dataframe(filtered_df, use_container_width=True)

    # 통계 정보
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 문서 수", len(docs_df))
    with col2:
        st.metric("카테고리 수", len(docs_df["category"].unique()))
else:
    st.warning("색인된 문서가 없습니다.")
