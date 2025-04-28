from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os

CHROMA_DB_PATH = "data/vector_db"


def get_vectordb():
    """
    색인된 문서의 벡터 데이터베이스에 접근
    """
    # 벡터 DB가 존재하는지 확인
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"벡터 데이터베이스를 찾을 수 없습니다: {CHROMA_DB_PATH}"
        )

    # Chroma DB 로드
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

    return vectordb


def search_documents(query, n_results=3, category=None):
    """
    벡터 데이터베이스에서 관련 문서 검색

    Args:
        query: 검색 쿼리
        n_results: 반환할 결과 수
        category: 특정 카테고리(폴더)로 필터링

    Returns:
        documents: 검색된 Document 객체 리스트
    """
    vectordb = get_vectordb()

    # 카테고리 필터링이 있는 경우
    if category and category != "전체":
        results = vectordb.similarity_search(
            query, k=n_results, filter={"category": category}
        )
    else:
        results = vectordb.similarity_search(query, k=n_results)

    return results


def format_document_sources(docs):
    """
    검색된 문서 출처 정보 포맷팅

    Args:
        docs: Document 객체 리스트

    Returns:
        sources_str: 출처 정보 문자열
    """
    sources = []
    seen_sources = set()

    for doc in docs:
        if "filename" in doc.metadata and "category" in doc.metadata:
            source_str = f"- {doc.metadata['category']}/{doc.metadata['filename']}"
            if source_str not in seen_sources:
                sources.append(source_str)
                seen_sources.add(source_str)

    return "\n".join(sources)


# RAG 프롬프트 템플릿
RAG_PROMPT_TEMPLATE = """
질문에 답변하기 위해 다음 문맥을 사용하세요:

{context}

질문: {question}

답변:
"""


def get_rag_prompt():
    """
    RAG 응답 생성용 프롬프트 템플릿 반환
    """
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )
