from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

CHROMA_DB_PATH = "data/vector_db"

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

    docs = vectordb.get(where={"category": "interpretation"})
    total = len(docs["documents"])
    print(f"총 {total}개의 벡터 문서가 저장되어 있습니다.\n")
    for i in range(min(5, total)):
        print(f"문서 {i + 1}:")
        print("내용:", docs["documents"][i][:300], "...\n")  # 300자만 미리보기
        print("메타데이터:", docs["metadatas"][i])
        print("=" * 40)
