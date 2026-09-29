# import pandas as pd
#
# def main():
#     # 파일 읽기
#     df = pd.read_excel("전처리결과(3)_제거.xlsx", engine='openpyxl')
#
#     # 'review' 열에서 문장 길이가 10-500 사이인 행만 선택 (길이가 아니라 단어인듯)
#     df_filtered = df[df['review'].apply(lambda x: len(x.split()) >= 5 and len(x.split()) <= 500)]
#
#     # 새로운 엑셀 파일로 저장
#     df_filtered.to_excel("전처리결과(3)_문장길이.xlsx", index=False)
#
# if __name__ == "__main__":
#     main()

## 이거는 단어읭 개수로 한거임

# ====================================================

# import pandas as pd
#
# # 엑셀 파일 읽기
# df = pd.read_excel("전처리결과(3)_제거.xlsx", engine='openpyxl')
#
# # 각 리뷰의 길이 계산하고 데이터프레임 필터링
# df['length'] = df['review'].str.len()
# filtered_df = df[(df['length'] >= 5) & (df['length'] <= 500)]
#
# # 필터링된 데이터를 새로운 엑셀 파일로 저장
# filtered_df.reset_index(drop=True, inplace=True)
# filtered_df.to_excel("전처리결과(3)_문장길이.xlsx", index=False)

## 이게 문장 길이


# # ======================================================================
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Excel 파일을 읽어옵니다.
# file_path = '전처리결과(3)_문장길이.xlsx'
# data = pd.read_excel(file_path)
#
# # 'review' 열의 글자 수를 계산하여 새로운 열에 추가합니다.
# data['글자수'] = data['review'].apply(lambda x: len(str(x)))
#
# # 글자 수별 review 개수를 계산합니다.
# review_count_by_length = data['글자수'].value_counts().sort_index()
#
# # 그래프를 그립니다.
# plt.figure(figsize=(15, 6))
# plt.scatter(
#     review_count_by_length.index,
#     review_count_by_length.values,
#     color='#FA9CC5',  # 색상 변경
#     s=3,  # 점의 크기 조절
#     alpha=0.7
# )
# plt.title('Review 글자 수별 개수')
# plt.xlabel('Review 글자 수')
# plt.ylabel('개수')
# plt.grid(True)
# plt.show()
#
# # =======================================================
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 엑셀 파일 읽기
# df = pd.read_excel("전처리결과(3)_문장길이.xlsx", engine='openpyxl')
#
# # review의 길이 계산
# df['length'] = df['review'].apply(len)
#
# # 히스토그램 그리기
# plt.figure(figsize=(10, 6))
# plt.hist(df['length'], bins=range(0, 500, 10), color='#FA9CC5', edgecolor='black')
# plt.title('Distribution of number of reviews by review length')
# plt.xlabel('Review Length')
# plt.ylabel('Number of reviews')
# plt.grid(True)
#
# plt.show()

# =============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 엑셀 파일 읽기
df = pd.read_excel("전처리결과(3)_문장길이.xlsx", engine='openpyxl')

# KDE 플롯 그리기 (선 그래프로, 안을 채우도록 설정)
plt.figure(figsize=(10, 6))
sns.kdeplot(df['length'], shade=True, color='#FA9CC5')
plt.title('Distribution of number of reviews by review length')
plt.xlabel('Review Length')
plt.ylabel('Number of reviews')
plt.grid(True)

# x축 범위 설정 (0부터 100까지)
plt.xlim(0, 500)

plt.show()






