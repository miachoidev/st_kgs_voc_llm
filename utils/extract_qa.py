import re
import fitz  # PyMuPDF
import csv
import os
import pathlib

# 데이터 폴더 경로 설정
data_dir = "data"
qa_dir = os.path.join(data_dir, "qa")
output_dir = os.path.join(data_dir, "qa_csv")

# 출력 디렉토리가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# PDF 파일 목록 가져오기
pdf_files = [f for f in os.listdir(qa_dir) if f.endswith(".pdf")]

# 각 PDF 파일 처리
for pdf_file in pdf_files:
    pdf_path = os.path.join(qa_dir, pdf_file)

    # 출력 파일 이름 생성 (PDF 파일명과 동일하게)
    output_filename = os.path.splitext(pdf_file)[0] + ".csv"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Processing: {pdf_file}")

    # PDF 로드
    doc = fitz.open(pdf_path)

    # 모든 페이지에서 텍스트 추출
    text = ""
    for page in doc:
        text += page.get_text()

    # Split entries using "고객명" 시작 라인을 기준으로 나누기
    entries = re.split(r"\n고객명", text)
    entries = [
        "고객명" + e for e in entries if e.strip() and "제목" in e and "답변내용" in e
    ]

    qa_pairs_separated = []

    for entry in entries:
        try:
            # '제목\n...처리부서' 사이의 블록만 추출 (처리부서 라인은 제외)
            match = re.search(r"(제목\s*\n.*?)(?=\n처리부서)", entry, re.DOTALL)
            if not match:
                continue
            question_block = match.group(1).strip()

            # 제목 추출
            title_match = re.search(r"제목\s*\n(.*)", question_block)
            title = title_match.group(1).strip() if title_match else ""

            # 질의내용 추출: 제목 이후 줄부터 마지막 1줄 제거
            q_lines = question_block.splitlines()[
                1:-1
            ]  # 제목 라인 제외, 마지막 한 줄 제거
            question_body = " ".join([line.strip() for line in q_lines if line.strip()])

            # 답변 추출: '우리공사'부터 끝까지, 마지막 1줄 제거
            answer_match = re.search(r"(우리공사.*)", entry, re.DOTALL)
            if not answer_match:
                continue
            answer_raw = answer_match.group(1)
            answer_lines = answer_raw.strip().splitlines()[:-1]  # 마지막 1줄 제거
            answer_clean = " ".join(
                [line.strip() for line in answer_lines if line.strip()]
            )

            qa_pairs_separated.append([title, question_body, answer_clean])
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue

    # CSV 파일 저장
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["제목", "질문", "답변"])
        writer.writerows(qa_pairs_separated)

    print(f"Saved: {output_path} with {len(qa_pairs_separated)} QA pairs")

print("모든 PDF 파일 처리 완료")
