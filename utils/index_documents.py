import os
import glob
from typing import List, Dict, Any
import docx2txt
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import pandas as pd
from dotenv import load_dotenv
import tempfile
from io import StringIO
from langchain.docstore.document import Document as LangchainDocument
import olefile
import unicodedata

# 환경 변수 로드
load_dotenv()

# 색인할 파일 유형 정의
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".docx", ".hwp"]
RAG_DATA_PATH = "data/rag"
CHROMA_DB_PATH = "data/vector_db"

# 카테고리별 영문 컬렉션 이름 매핑
CATEGORY_COLLECTION_MAP = {
    "법령": "law",
    "수수료": "fee",
    "체적거래제": "volume_trade",
    "기타": "etc",
}


class HwpLoader:
    """한글(HWP) 파일 로더 클래스 (olefile 사용)"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[LangchainDocument]:
        """HWP 파일을 로드하여 텍스트 추출 (olefile 사용)"""
        try:
            # olefile을 사용하여 HWP 파일 열기
            ole = olefile.OleFileIO(self.file_path)

            # PrvText 스트림에서 인코딩된 텍스트 읽기
            encoded_text = ole.openstream("PrvText").read()

            # utf-16으로 디코딩
            text = encoded_text.decode("utf-16")

            # 문서 객체 생성 및 반환
            return [
                LangchainDocument(
                    page_content=text, metadata={"source": self.file_path}
                )
            ]
        except Exception as e:
            print(f"HWP 파일 처리 오류: {e}")
            # 변환 실패 시 빈 문서 반환
            return [
                LangchainDocument(
                    page_content="[HWP 파일 변환 실패]",
                    metadata={"source": self.file_path},
                )
            ]


class CustomPDFLoader:
    """PDF 파일을 단일 문서로 로드하는 클래스"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[LangchainDocument]:
        """PDF 파일을 하나의 문서로 로드"""
        try:
            # 기존 PyPDFLoader 사용
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()

            # 모든 페이지 내용을 하나로 합치기
            combined_text = "\n\n".join([doc.page_content for doc in docs])

            # 단일 문서 반환
            return [
                LangchainDocument(
                    page_content=combined_text, metadata={"source": self.file_path}
                )
            ]
        except Exception as e:
            print(f"PDF 파일 처리 오류: {e}")
            # 변환 실패 시 빈 문서 반환
            return [
                LangchainDocument(
                    page_content="[PDF 파일 변환 실패]",
                    metadata={"source": self.file_path},
                )
            ]


def get_file_loader(file_path: str):
    """파일 확장자에 맞는 로더 반환"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return CustomPDFLoader(file_path)  # 커스텀 PDF 로더 사용
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".hwp":
        # 커스텀 HwpLoader 사용
        return HwpLoader(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def load_documents_from_directory(directory: str) -> List[Document]:
    """디렉토리에서 문서 로드 및 메타데이터 추가"""
    documents = []

    # 카테고리(폴더명) 가져오기
    category_kr = os.path.basename(directory)
    category_kr = category_kr.strip()
    category_kr = unicodedata.normalize("NFC", category_kr)
    print(f"category_kr: [{repr(category_kr)}]")  # 디버깅용

    category = CATEGORY_COLLECTION_MAP.get(category_kr, "etc")
    print(f"category: [{category_kr}] → [{category}]")

    # 지원하는 모든 확장자의 파일 찾기
    for ext in SUPPORTED_EXTENSIONS:
        file_pattern = os.path.join(directory, f"*{ext}")
        for file_path in glob.glob(file_pattern):
            try:
                loader = get_file_loader(file_path)
                file_docs = loader.load()

                # 메타데이터 추가 (영문 카테고리, 파일명)
                filename = os.path.basename(file_path)
                for doc in file_docs:
                    doc.metadata["category"] = category  # 영문 컬렉션명으로 저장
                    doc.metadata["filename"] = filename
                    doc.metadata["source"] = file_path

                documents.extend(file_docs)
                print(f"로드 완료: {file_path}")
            except Exception as e:
                print(f"파일 로드 실패: {file_path}, 오류: {str(e)}")

    return documents


def index_all_documents():
    """모든 문서 색인 및 벡터 저장소 생성"""
    all_documents = []
    # categories = []

    # RAG_DATA_PATH 디렉토리 내의 모든 하위 디렉토리 확인
    for item in os.listdir(RAG_DATA_PATH):
        dir_path = os.path.join(RAG_DATA_PATH, item)
        if os.path.isdir(dir_path):
            print(f"디렉토리 처리 중: [{dir_path}]")
            # categories.append(item)
            docs = load_documents_from_directory(dir_path)
            all_documents.extend(docs)

    print(f"총 {len(all_documents)} 문서 로드됨")

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunked_documents = text_splitter.split_documents(all_documents)
    print(f"총 {len(chunked_documents)} 청크로 분할됨")

    # Embeddings 생성 및 Chroma DB에 저장
    embeddings = OpenAIEmbeddings()
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    # 색인된 문서 목록을 CSV 파일로 저장
    docs_info = []
    for doc in all_documents:
        docs_info.append(
            {"category": doc.metadata["category"], "filename": doc.metadata["filename"]}
        )

    # 중복 제거 및 CSV 저장
    df = pd.DataFrame(docs_info).drop_duplicates()
    df.to_csv("data/indexed_documents.csv", index=False)

    # vectordb.persist()  # 이 줄을 삭제 또는 주석 처리
    print(f"벡터 DB가 {CHROMA_DB_PATH}에 저장되었습니다.")

    return df


if __name__ == "__main__":
    index_all_documents()
