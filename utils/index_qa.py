import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


CHROMA_QA_PATH = "data/vector_qa_db"


def load_data(csv_path):
    """CSV 파일에서 민원 데이터를 로드합니다."""
    df = pd.read_csv(csv_path)
    return df


def create_chroma_index(df, collection_name, persist_directory):
    """
    민원 데이터를 Chroma DB에 색인합니다.
    색인 전략:
    1. 제목+질문을 함께 임베딩하여 검색용 텍스트로 사용
    2. 메타데이터에 원본 제목, 질문, 답변을 저장하여 검색 결과에서 접근 가능하도록 함
    """
    # 임베딩 모델 설정
    embeddings = OpenAIEmbeddings()

    # 데이터 준비 - LangChain Document 형식으로 변환
    documents = []

    for idx, row in df.iterrows():
        # 제목과 질문을 합쳐서 임베딩용 텍스트로 사용
        doc_text = f"{row['제목']} {row['질문']}"

        # 메타데이터에 원본 필드 모두 저장
        metadata = {
            "title": row["제목"],
            "question": row["질문"],
            "answer": row["답변"],
            "id": f"doc_{idx}",
        }

        # Document 객체 생성
        doc = Document(page_content=doc_text, metadata=metadata)
        documents.append(doc)

    # Chroma 벡터스토어 생성
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"총 {len(documents)}개의 민원 데이터를 색인했습니다.")
    return vectordb


def query_similar_complaints(vectordb, query_text, n_results=5):
    """
    새로운 민원 텍스트와 가장 유사한 기존 민원을 검색합니다.
    query_text: 제목이나 질문 또는 둘 다 포함된 텍스트
    """
    similar_docs = vectordb.similarity_search_with_score(query_text, k=n_results)

    # 결과 변환
    results = []
    for doc, score in similar_docs:
        results.append(
            {"content": doc.page_content, "metadata": doc.metadata, "similarity": score}
        )

    return results


if __name__ == "__main__":
    # 설정
    csv_path = "data/qa_csv/merged_qa.csv"
    collection_name = "complaints_qa"
    # persist_directory = "./chroma_db"
    os.makedirs(CHROMA_QA_PATH, exist_ok=True)

    # 데이터 로드
    df = load_data(csv_path)
    print(f"민원 데이터 로드 완료: {len(df)}개의 항목")

    # Chroma 색인 생성
    vectordb = create_chroma_index(df, collection_name, CHROMA_QA_PATH)
