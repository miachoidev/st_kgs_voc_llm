import streamlit as st
import pandas as pd

# # 페이지 기본 설정
# st.set_page_config(page_title="민원 자동응답 시스템", layout="wide")

pages = [
    st.Page("home_page.py", title="소개"),
    st.Page("voc_page.py", title="민원 자동응답"),
    # st.Page("rag_page.py", title="RAG 문서 관리"),
    st.Page("eval_page.py", title="민원 자동응답 성능평가"),
    # "홈": [
    #     st.Page("home_page.py", title="홈"),
    # ],
    # "민원 자동응답 서비스": [
    #     st.Page("voc_page.py", title="민원 자동응답"),
    #     st.Page("rag_page.py", title="RAG 문서 관리"),
    # ],
    # "평가": [
    #     st.Page("eval_page.py", title="민원 자동응답 성능평가"),
    # ],
]

pg = st.navigation(pages)
pg.run()
