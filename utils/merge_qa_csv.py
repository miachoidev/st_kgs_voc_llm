import pandas as pd
import os
import glob

# CSV 파일이 있는 디렉토리 경로
data_dir = "data/qa_csv"

# CSV 파일 목록 가져오기
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
print(f"병합할 CSV 파일 목록: {csv_files}")

# 빈 데이터프레임 생성
merged_df = pd.DataFrame()

# 각 CSV 파일을 읽어서 하나의 데이터프레임으로 병합
for file in csv_files:
    df = pd.read_csv(file)
    print(f"{file} - 행 수: {len(df)}")
    merged_df = pd.concat([merged_df, df], ignore_index=True)

# 답변 내용에 '첨부파일'가 포함된 행 제거
merged_df = merged_df[~merged_df["답변"].str.contains("첨부파일")]

# 답변 내용에 '유선'가 포함된 행 제거
merged_df = merged_df[~merged_df["답변"].str.contains("유선")]

# 답변 내용에 '답변은 질의번호 93561에 함께 올려두었으니'가 포함된 행 제거
merged_df = merged_df[
    ~merged_df["답변"].str.contains("답변은 질의번호 93561에 함께 올려두었으니")
]

# 질문 내용에 '첨부파일'가 포함된 행 제거
merged_df = merged_df[~merged_df["질문"].str.contains("첨부파일")]

# 질문 내용에 '유선'가 포함된 행 제거
merged_df = merged_df[~merged_df["질문"].str.contains("유선")]

# 병합된 CSV 파일 저장
output_file = "data/qa_csv/merged_qa.csv"
merged_df.to_csv(output_file, index=False)

print(f"병합 완료! 전체 {len(merged_df)}개 행이 {output_file}에 저장되었습니다.")
