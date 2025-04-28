import streamlit as st
import pandas as pd

st.markdown(
    """
    <h1 style='text-align: center; color: #1E88E5;'>가스안전공사 민원 자동응답 시스템</h1>
    <h3 style='text-align: center; color: #424242;'>LLM 기반 민원처리 효율화 솔루션</h3>
    """,
    unsafe_allow_html=True,
)

# 구분선
st.divider()

# 프로젝트 소개
st.subheader("🎯 시스템 목적")
st.markdown("""
    한국가스안전공사의 민원(VOC) 처리 효율화를 위해 개발된 AI 기반 솔루션으로, 자연어 처리 기술을 활용하여:
    
    - 📝 **민원 답변 초안 자동 생성**: 담당자의 업무 효율성 향상
    - 🔍 **유사 과거 민원 검색**: 일관성 있는 응대 지원
    - ⏱️ **처리 시간 단축**: 민원인 만족도 증가
    - 🧠 **기관 지식 활용 극대화**: 축적된 데이터 기반 의사결정
    """)

# 2단 레이아웃 (데이터와 기술)
col1, col2 = st.columns(2)

with col1:
    st.subheader("💻 핵심 기술")
    st.markdown("""
        - **LLM(대규모 언어 모델)**: 자연어 이해 및 생성 GPT-4o-mini
        - **벡터 데이터베이스**: Chroma DB, embedding: text-embedding-3-small
        - **RAG(검색 증강 생성)**: 관련 문서 검색 기반 응답 생성, retriever: EnsembleRetriever(BM25 + vector) 
        - **유사 민원 검색**: 유사 민원 검색 결과 제공(similarity_search)
        - **평가 시스템**: LLM Judge 방식의 성능 평가
        - **프롬프트 엔지니어링**: 민원답변 특성에 맞는 프롬프트 엔지니어링, 평가용 프롬프트 엔지니어링
        """)

with col2:
    st.subheader("📊 활용 데이터")
    st.markdown("""
        - **과거 민원 데이터**: 수천 건의 처리 완료된 VOC 데이터
        - **법률 및 규정**: 가스 관련 법률, 시행령, 시행규칙
        - **내부 지침**: 한국가스안전공사 업무 지침 및 매뉴얼
        """)


# 시스템 구성도
st.subheader("🔄 시스템 구성도")

# 간단한 시스템 흐름도
st.graphviz_chart("""
digraph {
    node [shape=box, style=filled, color=lightblue, fontname="나눔고딕"];
    민원접수 [label="민원 접수"];
    RAG [label="RAG 법률/규정 검색"];
    SEARCH [label="유사민원 검색"];
    LLM [label="LLM 답변 생성"];
    유사민원 [label="유사 민원 검색 결과 제공"];
    응답 [label="민원 응답 초안 생성"];
    

    민원접수 -> RAG;
    민원접수 -> SEARCH;
    SEARCH -> 유사민원;
    RAG -> LLM;
    LLM -> 응답;

    {rank=same; RAG SEARCH}
    
}
""")


# 푸터
st.divider()
st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: small;'>
    © 2025 한국가스안전공사 민원 자동응답 시스템  PoC | 업무 프로젝트를 포트폴리오 프로젝트로 재구성하였습니다.<br>
    </p>
    <p style='text-align: center; color: gray; font-size: small;'>
    <b>✉️ e-mail:</b> mia.choi.dev@gmail.com | <b>✨ 포트폴리오 Notion:</b>  https://mia-choi.github.io/
    </p>
    """,
    unsafe_allow_html=True,
)
