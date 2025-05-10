import streamlit as st
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from openai import OpenAI
from utils.langgraph_rag import run_rag
from dotenv import load_dotenv
import random

# .env 파일 로드
load_dotenv()

# OpenAI API 키 획득
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide")

st.title("LLM-as-a-judge evaluation")
st.info(
    "정답셋과 모델 응답을 비교하여 평가합니다. 평가모델(judge)은 GPT-4 Turbo를 사용합니다."
)

# 평가 지표 설명 추가
st.markdown("""
### 평가 지표
- **근거 일치 (40점)**: 실제 정답에서 사용된 법령/규정 근거를 정확히 찾아 사용했는지
- **신뢰성 (30점)**: 실제 답변과 무관하게 검색한 문서를 기반으로 올바르게 답변했는지
- **완전성 (20점)**: 실제 답변의 질의 요지를 빠짐없이 뽑아냈는지. 각 질의요지별 답변을 작성했는지.(답변내용의 정확성은 제외, 질문 파악성능에 초점.)
- **구조 준수 (10점)**: <질의요지>, <답변내용> 형식을 정확히 따랐는지, 고정 인사말이나 답변자정보는 제외하고 평가
""")

# CSV 파일 경로
csv_path = "data/qa_csv/merged_qa.csv"
results_dir = "data/evaluation_results"
os.makedirs(results_dir, exist_ok=True)

# 전역 변수 초기화
if "ready_df" not in st.session_state:
    st.session_state.ready_df = None

if "has_llm_answers" not in st.session_state:
    st.session_state.has_llm_answers = False

if "has_evaluation" not in st.session_state:
    st.session_state.has_evaluation = False


def load_random_samples(n_samples):
    """지정된 수만큼 랜덤 샘플 로드"""
    try:
        full_df = pd.read_csv(csv_path)
        # 전체 데이터에서 n_samples 개수만큼 랜덤 샘플링
        # if n_samples >= len(full_df):
        #     n_samples = len(full_df)

        # sample_indices = random.sample(range(len(full_df)), n_samples)
        # df = full_df.loc[sample_indices].copy().reset_index(drop=True)
        # 전체데이터에서 순서대로 n_samples 개만 로드
        df = full_df.head(n_samples)

        # 필요한 열 추가
        if "llm_answer" not in df.columns:
            df["llm_answer"] = ""
        if "evaluation" not in df.columns:
            df["evaluation"] = ""
        if "score" not in df.columns:
            df["score"] = 0.0
        if "accuracy" not in df.columns:
            df["accuracy"] = 0.0
        if "completeness" not in df.columns:
            df["completeness"] = 0.0
        if "evidence" not in df.columns:
            df["evidence"] = 0.0
        if "structure" not in df.columns:
            df["structure"] = 0.0

        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None


# LLM 응답을 GPT-4 Turbo로 평가
def evaluate_with_gpt4(question, real_answer, llm_answer, retrieved_documents):
    try:
        client = OpenAI()
        messages = [
            {
                "role": "system",
                "content": """
            당신은 한국 가스안전공사의 민원 답변 평가 전문가입니다.
            실제 답변과 비교하여 총점(100점 만점)을 기준으로 다음 4가지 기준으로 llm 답변을 평가하세요:
            
            1. 근거 일치 (40점): 실제 정답에서 사용된 법령/규정 등 문서와 동일한 문서를 찾았는지.
            세부 규칙 1) 참고한 문서의 제목이 완벽하지 않아도 비슷한 제목의 문서는 반드시 일치한것으로 간주합니다.(세부 장절 항목은 제외하고 동일한 문서만 찾아도 35점).
            - 예시1) 「액화석유가스의 안전관리 및 사업법」 과 「액화석유가스의 안전관리 및 사업법 시행규칙」 은 동일한 문서임(35점)
            - 예시2) KGS FU433_231107.pdf 문서는 실제 답변에서 언급된 KGS FU432 2.3.3.4(3) 문서와 동일한 문서임(35점), KGS FU433와 비슷하게 KGS+코드값 패턴의 제목 문서는 핵심 코드만 같으면 같은 문서입니다.
            세부 규칙 2) 법령 문서의 경우 장, 절, 관, 조, 항을 정확하게 제시했다면 추가로 5점을 주고 40점 만점으로 평가. 
            2. 완전성 (20점): 실제 답변과 비교했을때 <질의요지>에 핵심 질문의 내용을 비슷하게 파악했는지.
            3. 신뢰성 (30점) : 실제 답변과 무관하게 모델이 검색한 문서의 내용을 기반으로 답했는지.(실제 답변과 달라도 검색문서를 기반으로 답변했으면 만점)
            4. 구조 준수 (10점): <질의요지>, <답변내용> 말머리를 정확히 표기하여 작성하였는지(고정 인사말이나 답변자정보 및 세부 구조는 제외하고 저 말머리만 준수하면 만점.)
            
            각각 항목별 점수와 한문장 이내의 간단한 평가 이유를 작성하세요.
            각 4가지 기준 점수는 반드시 '정확성: 20점'과 같은 형식으로 작성해야 합니다. '정확성(10/50)' 이런 형식은 안됩니다.
            
            """,
            },
            {
                "role": "user",
                "content": f"""
            [질문]
            {question}
            
            [실제 답변]
            {real_answer}
            
            [LLM 답변]
            {llm_answer}
            
            [검색 문서]
            {retrieved_documents}
            
            위 내용을 위 기준에 따라 평가해주세요.
             """,
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )

        evaluation_text = response.choices[0].message.content

        # 점수 추출 (정규식으로 100점 만점과 각 항목별 점수 추출)
        import re

        # total_score_match = re.search(r"총점[^\d]*(\d+)점", evaluation_text)
        accuracy_match = re.search(r"신뢰성[^\d]*(\d+)점", evaluation_text)
        completeness_match = re.search(r"완전성[^\d]*(\d+)점", evaluation_text)
        evidence_match = re.search(r"근거 일치[^\d]*(\d+)점", evaluation_text)
        structure_match = re.search(r"구조 준수[^\d]*(\d+)점", evaluation_text)

        accuracy = float(accuracy_match.group(1)) if accuracy_match else 0.0
        completeness = float(completeness_match.group(1)) if completeness_match else 0.0
        evidence = float(evidence_match.group(1)) if evidence_match else 0.0
        structure = float(structure_match.group(1)) if structure_match else 0.0
        total_score = accuracy + completeness + evidence + structure

        return evaluation_text, total_score, accuracy, completeness, evidence, structure

    except Exception as e:
        return f"평가 중 오류 발생: {str(e)}", 0.0, 0.0, 0.0, 0.0, 0.0


# 병렬 처리 함수
def process_question(question, index):
    try:
        # print(f"질문 처리 시작: {question[:50]}...")
        result = run_rag(question)
        answer = result["answer"]
        # print(f"질문 처리 완료. 결과: {answer[:100]}...")
        return index, answer, result["retrieved_documents"]
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        print(f"오류 발생: {str(e)}")
        import traceback

        traceback_str = traceback.format_exc()
        print(f"상세 오류: {traceback_str}")
        return index, error_msg


# 평가할 데이터 갯수 지정
sample_size = st.number_input(
    "평가할 데이터 갯수 지정",
    min_value=1,
    max_value=100,  # 최대값을 100으로 제한
    value=10,
)

# 응답 생성 버튼
if st.button("응답 생성", use_container_width=True):
    # 지정된 갯수만큼 랜덤 샘플 로드
    df = load_random_samples(sample_size)
    if df is not None:
        # 진행 상황을 표시할 progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 질문 리스트 생성
        questions = [(idx, df.loc[idx, "질문"]) for idx in range(len(df))]
        total = len(questions)

        with st.spinner("LLM 응답을 생성하는 중..."):
            start_time = time.time()

            # ThreadPoolExecutor를 사용한 병렬 처리
            with ThreadPoolExecutor(max_workers=4) as executor:
                # 작업 제출
                futures = [
                    executor.submit(process_question, question, idx)
                    for idx, question in questions
                ]

                # 결과 수집
                completed = 0
                for future in as_completed(futures):
                    idx, answer, retrieved_documents = future.result()

                    # 데이터프레임 업데이트
                    df.at[idx, "llm_answer"] = answer

                    # Document 객체에서 필요한 정보만 추출
                    processed_docs = []
                    for doc in retrieved_documents:
                        doc_info = {
                            "id": getattr(doc, "id", None),
                            "file_name": doc.metadata.get("filename", ""),
                            # "page_content": doc.page_content[:200] + "..."
                            "page_content": doc.page_content,
                            # if len(doc.page_content) > 200
                            # else doc.page_content,
                        }
                        processed_docs.append(doc_info)

                    df.at[idx, "retrieved_documents"] = json.dumps(
                        processed_docs, ensure_ascii=False
                    )
                    # 진행 상황 업데이트
                    completed += 1
                    progress = completed / total
                    progress_bar.progress(progress)
                    status_text.text(f"처리 중... {completed}/{total} 완료")

            end_time = time.time()

            # 결과 표시
            st.success(f"모든 응답 생성 완료! 소요 시간: {end_time - start_time:.2f}초")

            # 상태 업데이트
            st.session_state.ready_df = df
            st.session_state.has_llm_answers = True

# 데이터프레임 표시
if st.session_state.has_llm_answers and st.session_state.ready_df is not None:
    st.subheader("LLM 응답 결과")
    show_df = st.session_state.ready_df[
        ["제목", "질문", "답변", "llm_answer", "retrieved_documents"]
    ]
    st.dataframe(
        show_df.rename(
            columns={
                "제목": "민원 제목",
                "질문": "민원 내용",
                "답변": "실제 답변",
                "llm_answer": "LLM 답변",
                "retrieved_documents": "검색 문서",
            }
        ),
        use_container_width=True,
        height=400,
        column_config={
            "민원 제목": st.column_config.TextColumn(width="small"),
            "민원 내용": st.column_config.TextColumn(width="small"),
            "실제 답변": st.column_config.TextColumn(width="small"),
            "LLM 답변": st.column_config.TextColumn(width="medium"),
            "검색 문서": st.column_config.TextColumn(width="medium"),
        },
    )

    # 평가하기 버튼
    if st.button("평가하기", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("평가 중..."):
            start_time = time.time()

            # 데이터프레임에 접근
            df = st.session_state.ready_df

            # 병렬 처리 함수
            def process_evaluation(idx):
                row = df.loc[idx]
                evaluation_result = evaluate_with_gpt4(
                    row["질문"],
                    row["답변"],
                    row["llm_answer"],
                    row["retrieved_documents"],
                )

                return idx, evaluation_result

            # ThreadPoolExecutor를 사용한 병렬 처리
            with ThreadPoolExecutor(
                max_workers=2
            ) as executor:  # 병렬 처리 수를 2로 줄임
                # 작업 제출
                futures = [
                    executor.submit(process_evaluation, idx) for idx in range(len(df))
                ]

                # 결과 수집
                completed = 0
                total = len(df)
                for future in as_completed(futures):
                    (
                        idx,
                        (
                            evaluation_text,
                            score,
                            accuracy,
                            completeness,
                            evidence,
                            structure,
                        ),
                    ) = future.result()

                    df.at[idx, "evaluation"] = evaluation_text
                    df.at[idx, "score"] = score
                    df.at[idx, "accuracy"] = accuracy
                    df.at[idx, "completeness"] = completeness
                    df.at[idx, "evidence"] = evidence
                    df.at[idx, "structure"] = structure

                    # 진행 상황 업데이트
                    completed += 1
                    progress = completed / total
                    progress_bar.progress(progress)
                    status_text.text(f"평가 중... {completed}/{total} 완료")

                    # rate limit 방지를 위한 지연 시간 추가
                    if completed < total:  # 마지막 항목이 아닌 경우에만 대기
                        time.sleep(10)  # 10초 대기

            end_time = time.time()

            # 결과 표시
            st.success(f"모든 평가 완료! 소요 시간: {end_time - start_time:.2f}초")

            # 상태 업데이트
            st.session_state.ready_df = df
            st.session_state.has_evaluation = True

# 평가 결과 표시
if st.session_state.has_evaluation and st.session_state.ready_df is not None:
    st.subheader("평가 결과")

    # 평가 결과 데이터프레임
    df = st.session_state.ready_df
    eval_df = df[
        [
            "제목",
            "score",
            "accuracy",
            "completeness",
            "evidence",
            "structure",
            "evaluation",
        ]
    ]
    st.dataframe(
        eval_df.rename(
            columns={
                "제목": "민원 제목",
                "score": "총점",
                "accuracy": "정확성",
                "completeness": "완전성",
                "evidence": "근거 일치",
                "structure": "구조 준수",
                "evaluation": "평가 내용",
            }
        ),
        use_container_width=True,
        height=400,
        column_config={
            "민원 제목": st.column_config.TextColumn(width="small"),
            "총점": st.column_config.NumberColumn(width="small"),
            "신뢰성": st.column_config.NumberColumn(width="small"),
            "완전성": st.column_config.NumberColumn(width="small"),
            "근거 일치": st.column_config.NumberColumn(width="small"),
            "구조 준수": st.column_config.NumberColumn(width="small"),
            "평가 내용": st.column_config.TextColumn(width="medium"),
        },
    )

    # 평균 점수 계산
    avg_score = df["score"].mean()
    avg_accuracy = df["accuracy"].mean()
    avg_completeness = df["completeness"].mean()
    avg_evidence = df["evidence"].mean()
    avg_structure = df["structure"].mean()

    # 평균 점수 표시
    st.subheader("평가 점수 요약")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("총점 평균", f"{avg_score:.1f}점")
        st.caption("전체 점수 평균 (100점 만점)")

    with col2:
        st.metric("신뢰성 평균", f"{avg_accuracy:.1f}점")
        st.caption("정답과 무관하게 찾은 문서를 기반으로 답변했는지 (50점 만점)")

    with col3:
        st.metric("완전성 평균", f"{avg_completeness:.1f}점")
        st.caption("질문의 모든 핵심을 빠짐없이 다루었는지 (20점 만점)")

    with col4:
        st.metric("근거 일치 평균", f"{avg_evidence:.1f}점")
        st.caption(
            "실제 정답에서 사용된 법령/규정 근거를 정확히 찾아 사용했는지 (20점 만점)"
        )

    with col5:
        st.metric("구조 준수 평균", f"{avg_structure:.1f}점")
        st.caption("<질의요지>, <답변내용> 형식을 정확히 따랐는지 (10점 만점)")

    # 결과 CSV 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"{results_dir}/evaluation_results_{timestamp}.csv"
    result_df = df.copy()
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # CSV 다운로드 버튼
    with open(output_path, "rb") as f:
        csv_data = f.read()

    st.download_button(
        label="평가 결과 CSV 다운로드",
        data=csv_data,
        file_name=f"evaluation_results_{timestamp}.csv",
        mime="text/csv",
    )
