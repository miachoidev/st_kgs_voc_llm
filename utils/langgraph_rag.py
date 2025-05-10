from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pandas as pd
from typing import Dict, List, Tuple, TypedDict, Annotated, Sequence, Union
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
import numpy as np
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.output_parsers import StrOutputParser

# .env 파일 로드
load_dotenv()

# OpenAI API 키 획득
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 환경 변수에 설정되지 않았습니다.")

# 상수 정의
CHROMA_DB_PATH = "data/vector_db"
CSV_PATH = "data/qa_csv/merged_qa.csv"

# 카테고리 정의
CATEGORIES = ["법령", "수수료", "체적거래제", "기타"]

# 카테고리별 영문 컬렉션 이름 매핑
CATEGORY_COLLECTION_MAP = {
    "법령": "law",
    "수수료": "fee",
    "체적거래제": "volume_trade",
    "기타": "etc",
}


# 타입 정의
class AgentState(TypedDict):
    question: str
    categories: List[str]
    retrieved_documents: List[Document]
    answer: str
    sources: str


# 유틸리티 함수
def load_qa_data():
    """
    merged_qa.csv 파일에서 제목과 질문 데이터를 추출
    """
    df = pd.read_csv(CSV_PATH)
    return df[["제목", "질문"]]


def get_vectordb():
    """
    벡터 데이터베이스 접근
    """
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(
            f"벡터 데이터베이스를 찾을 수 없습니다: {CHROMA_DB_PATH}"
        )

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        # collection_name=collection_name,
    )

    return vectordb


def format_document_sources(docs):
    """
    검색된 문서 출처 정보 포맷팅
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


# 노드 함수
def classify_query(state: AgentState) -> AgentState:
    """
    질문을 분석하여 관련 카테고리를 식별하는 노드
    """
    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

    prompt = PromptTemplate.from_template("""
    당신은 한국가스안전공사 관련 사용자 질문을 4개 카테고리로 분류하는 전문가입니다. 사용자 질문에 답변하기 위해 어떤 종류의 문서를 참고해야할지 분류하세요.

1. **법령**  
: 가스안전 관련 법령(법률, 시행령, 시행규칙, 고시, 훈령, 예규)과 설치 및 검사 등에 필요한 기술기준(기술코드, 표준코드 등)을 함께 참고해야 하는 질문  
 액화석유가스,소형저장탱크,방폭전기기,가스시설 용접,가스시설 전기방식 ,주거용/상업용 가스보일러의 시설,설계,설치,비파괴시험검사기준에 대한 질문
(※ 실제 설치 방법, 검사 기준, 설비 요건, 규정 준수 여부 등)

2. **수수료**  
: 검사비, 수수료, 요금표, 부과 기준, 납부 방법 등 비용 관련 질문

3. **체적거래제**  
: 체적 거래제(가스 판매 방식, 거래 기준, 가격 책정 등)와 관련된 질문

4. **기타**  
: 위 1~3번에 명확히 해당되지 않는 모든 질문


## 카테고리별 핵심 키워드

### 1. 법령
- 법률, 시행령, 시행규칙, 고시, 훈령, 예규, 법적 의무, 규제 준수
- 기술코드, 표준코드, 품질코드, 안전코드, 가스설비 설치기준, 검사기준, 설계기준
- 설치 방법, 검사 절차, 설비 요건, 현장 설치 방식, 배관, 용접, 탱크 교체, 시설 변경

### 2. 수수료
- 검사비, 인증비용, 수수료 기준, 요금표, 납부 방법, 수수료 감면

### 3. 체적거래제
- 체적 거래 기준, 가스 거래 방법, 체적 거래 가격, 거래 방식

### 4. 기타
- 위의 3개 카테고리에 해당되지 않는 모든 키워드

## 분류 지시사항

1. 사용자의 질문을 주의 깊게 읽으세요.
2. 질문에 포함된 키워드를 확인하세요.
3. 질문의 의도(정보 요청 vs 실제 설치/검사 방법 등)를 파악하세요.
4. 가장 적합한 카테고리를 선택하세요.
5. 여러 카테고리에 해당할 경우, 질문의 주된 목적에 따라 선택하세요.
6. '1~3번' 카테고리에 명확하게 해당되지 않으면 '4. 기타'로 분류하세요.
7. 분류 결과와 그 이유를 간략히 설명하세요.

    질문: {question}
    
    응답 형식: 카테고리 번호만 숫자로 답변하세요 (1~4)
    """)

    chain = prompt | llm
    result = chain.invoke({"question": state["question"]})
    print(f"classify_query result: {result.content}")

    # 결과에서 카테고리 번호 추출 - 숫자만 추출
    category_number = None
    for char in result.content:
        if char.isdigit() and 1 <= int(char) <= 7:
            category_number = int(char)
            break

    # 카테고리 번호에 따른 카테고리 매핑
    category_map = {
        1: "법령",
        2: "수수료",
        3: "체적거래제",
        4: "기타",
    }

    if category_number is None:
        category_kr = "기타"
    else:
        category_kr = category_map[category_number]

    category_en = CATEGORY_COLLECTION_MAP.get(category_kr, "etc")

    return {"question": state["question"], "categories": [category_en]}


def ensemble_search(
    state: AgentState,
) -> Dict[str, Union[str, List[str], List[Document]]]:
    """
    앙상블 검색: 벡터 검색과 BM25 검색을 결합
    """
    question = state["question"]
    categories = state["categories"]
    print(f"categories::: {categories}")
    all_docs = []
    vectordb = get_vectordb()
    # docs = vectordb.get(where={"category": {"$in": categories}})
    docs = vectordb.get()
    print(f"docs::: {len(docs)}")
    bm25_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(docs["documents"], docs["metadatas"])
    ]
    print(f"bm25_docs::: {len(bm25_docs)}")
    # vector_retriever = vectordb.as_retriever(
    #     search_kwargs={"k": 5, "filter": {"category": {"$in": categories}}}
    # )
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # BM25 리트리버 생성
    bm25_retriever = BM25Retriever.from_documents(bm25_docs, k=5)

    # 앙상블 리트리버 생성
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    result_docs = ensemble.get_relevant_documents(question)
    print(f"category_docs:::갯수::: {len(result_docs)}")

    # for category in categories:
    #     try:
    #         # 벡터 DB 접근
    #         vectordb = get_vectordb()

    #         # 문서 가져오기 (BM25용)

    #         result = vectordb.get(where={"category": category})
    #         # result = vectordb.get()
    #         all_category_docs = [
    #             Document(page_content=content, metadata=meta)
    #             for content, meta in zip(result["documents"], result["metadatas"])
    #         ]

    #         if not all_category_docs:
    #             continue

    #         # 벡터 리트리버 생성
    #         vector_retriever = vectordb.as_retriever(
    #             search_kwargs={"k": 5, "filter": {"category": category}}
    #         )

    #         # BM25 리트리버 생성
    #         bm25_retriever = BM25Retriever.from_documents(all_category_docs, k=5)

    #         # 앙상블 리트리버 생성
    #         ensemble = EnsembleRetriever(
    #             retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    #         )

    #         # 앙상블 검색 수행
    #         category_docs = ensemble.get_relevant_documents(question)
    #         print(f"category_docs:::갯수::: {len(category_docs)}")
    #         all_docs.extend(category_docs)

    #     except Exception as e:
    #         print(f"카테고리 '{category}' 검색 중 오류 발생: {e}")

    # 중복 제거
    unique_docs = []
    seen_contents = set()

    for doc in result_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)

    # 최대 5개 문서로 제한
    final_docs = unique_docs[:5]

    return {
        "question": state["question"],
        "categories": state["categories"],
        "retrieved_documents": final_docs,
    }


def generate_answer(state: AgentState, streaming: bool = False) -> AgentState:
    """
    최종 문서를 기반으로 답변 생성 (스트리밍 지원)

    Args:
        state: 현재 에이전트 상태
        streaming: 스트리밍 모드 활성화 여부

    Returns:
        업데이트된 에이전트 상태
    """
    question = state["question"]
    docs = state["retrieved_documents"]

    # 문서 컨텍스트 준비 - 제목과 내용을 구분하여 포맷팅
    context_parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("filename", f"문서 {i}")
        content = doc.page_content
        context_parts.append(f"### 문서 {i}: {title}\n{content}")

    context = "\n\n".join(context_parts)

    # 출처 정보 포맷팅
    sources = format_document_sources(docs)

    # LLM으로 답변 생성
    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, streaming=streaming)

    prompt = PromptTemplate.from_template("""
    당신은 가스안전 공사의 민원 답변 전문가 입니다. 다음 규정 및 법령 문서를 바탕으로 질문에 답변해주세요:

법령 및 참조 문서:
{context}

질문:
{question}

답변을 작성할 때 다음 가이드라인을 따라주세요:
1. 질문에 대한 명확한 해결방법보다는 위 법령 및 참조 문서에서 질문과 관련된 부분을 찾아 명확히 제시하세요.
2. 관련 법령이나 규정이 있다면 조문 번호까지 구체적으로 인용하세요.(예시 - "~법 제00조제0항에 따르면 ...")
3. 반드시 아래 형식에 맞추어 작성하세요. 답변예시를 참고하여 형식에 맞춰 작성하세요.:

[형식]
<질의요지>
(줄바꿈 필수. 질문의 핵심 요지를 파악하여 각 질문을 ㅇ 기호로 구분하여 작성하고 간결하고 짧게 요약)

<답변내용>
(질문 요지별 문서에 근거한 상세한 답변, 답변은 구분자 말머리 ㅇ 기호로 구분하며 줄바꿈 필수. )
ㅇ 질문 요지별 답변 내용
ㅇ 질문 요지별 답변 내용
---

[답변 예시]                                   
<질의요지>
ᄋ 질의 1. 소형저장탱크 주변의 주차장에 주차블럭 또는 경계책이 설치된 경우에도 소형저장 탱크 보호대를 설치하여야 하는지?
ᄋ 질의 2. 소형저장탱크와 주차설비 간의 거리에 따라 보호대 설치여부가 달라지는지?
                                          
<답변내용>
<질의 1에 대하여>
ᄋ KGS FU432 2.3.3.4(6)에서는 자동차 등에 의해 소형저장탱크가 손상을 받을 우려가 있는 경 우(주차장 내, 도로변 등)에는 다음 기준에 따라 보호대 등의 방호조치를 하도록 규정하고 있 습니다.
- 질의하신 주차블럭은 차량을 안전하게 주차할 수 있게 하기 위한 시설이며, KGS FU432 2.11.2에 따라 설치하는 경계책은 외부인의 출입을 막기 위한 시설이므로, 두 시설은 자동차 등의 충돌로부터 소형저장탱크를 보호할 수 없을 것으로 판단됩니다.
- 따라서, 상기 규정에 따라 소형저장탱크 사용시설의 주변에 주차블럭과 경계책이 설치된 경 우에도 보호대 등의 방호조치를 하여야 함을 알려드립니다.
<질의 2에 대하여>
ᄋ KGS FU432 2.3.3.4(6)(6-1) ~ (6-7)에서는 소형저장탱크 사용시설의 보호대 설치 기준을 규 정하고 있으며, 동 규정상에는 소형저장탱크와 주차설비 간의 거리에 따른 보호대 설치 여부에 대한 기준은 없음을 알려드립니다. 끝.
    """)

    chain = prompt | llm | StrOutputParser()

    if streaming:
        # 스트리밍 모드에서는 제너레이터를 반환 (voc_page.py에서 st.write_stream()으로 처리)
        # state에는 빈 문자열로 초기 설정 (이후 voc_page.py에서 스트리밍 후 재설정)
        stream_generator = chain.stream({"context": context, "question": question})

        return {
            "question": state["question"],
            "categories": state["categories"],
            "retrieved_documents": state["retrieved_documents"],
            "answer": stream_generator,  # 제너레이터 반환
            "sources": sources,
        }
    else:
        # 일반 모드에서는 전체 결과 반환
        result = chain.invoke({"context": context, "question": question})

        return {
            "question": state["question"],
            "categories": state["categories"],
            "retrieved_documents": state["retrieved_documents"],
            "answer": result,
            "sources": sources,
        }


# LangGraph 워크플로우 정의
def create_rag_workflow(streaming=False):
    """
    RAG 워크플로우 생성

    Args:
        streaming: 스트리밍 모드 활성화 여부
    """
    # 상태 그래프 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("ensemble_search", ensemble_search)
    # streaming 파라미터를 전달하기 위한 함수 래퍼 생성
    workflow.add_node(
        "generate_answer", lambda state: generate_answer(state, streaming=streaming)
    )

    # 엣지 연결 - 최신 LangGraph API 사용
    workflow.set_entry_point("classify_query")
    # workflow.set_entry_point("ensemble_search")

    # 노드 간 연결 정의
    workflow.add_edge("classify_query", "ensemble_search")
    workflow.add_edge("ensemble_search", "generate_answer")

    # 종료 노드 설정
    workflow.set_finish_point("generate_answer")

    # 컴파일
    return workflow.compile()


# 메인 함수
def run_rag(question: str, streaming: bool = False):
    """
    RAG 파이프라인 실행

    Args:
        question: 사용자 질문
        streaming: 스트리밍 모드 활성화 여부

    Returns:
        스트리밍 모드일 경우 (stream_generator, context_info) 튜플, 아닐 경우 결과 딕셔너리
    """
    try:
        # 워크플로우 생성 (streaming 파라미터 전달)
        rag_app = create_rag_workflow(streaming=streaming)

        # 초기 상태 설정 - TypedDict에 맞게 모든 필드 초기화
        initial_state = {
            "question": question,
            "categories": [],
            "retrieved_documents": [],
            "answer": "",
            "sources": "",
        }

        # 실행
        result = rag_app.invoke(initial_state)

        # 스트리밍 모드일 경우, 답변 스트림 제너레이터와 컨텍스트 정보를 함께 반환
        if streaming:
            # 스트림 제너레이터와 함께 문맥 정보도 반환
            context_info = {
                "retrieved_documents": result["retrieved_documents"],
            }

            print(f"result['answer']: {result['answer']}")

            return result["answer"], context_info  # 튜플 형태로 (스트림, 컨텍스트) 반환

        # 일반 모드일 경우, 기존처럼 전체 결과 반환
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "categories": result["categories"],
            "retrieved_documents": result["retrieved_documents"],
        }

    except Exception as error:
        import traceback

        print(f"RAG 실행 중 오류가 발생했습니다: {str(error)}")
        print(f"상세 오류: {traceback.format_exc()}")

        if streaming:

            def error_generator(err_msg):
                for char in err_msg:
                    yield char
                    import time

                    time.sleep(0.01)

            error_msg = f"답변 생성 중 오류가 발생했습니다: {str(error)}"
            # 오류 시에도 동일한 반환 형식 유지 (generator, empty_context)
            empty_context = {"sources": "", "categories": [], "retrieved_documents": []}
            return error_generator(error_msg), empty_context

        return {
            "answer": f"답변 생성 중 오류가 발생했습니다: {str(error)}",
            "sources": "",
            "categories": [],
            "retrieved_documents": [],
        }
