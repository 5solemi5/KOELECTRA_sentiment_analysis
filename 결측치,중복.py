# import pandas as pd
#
# # 엑셀 파일 불러오기
# excel_data = pd.read_excel('dataset_raw.xlsx')
#
# # 중복 제거
# excel_data.drop_duplicates(subset=['review'], inplace=True)
#
# # 결측치가 있는 행 제거
# excel_data.dropna(subset=['review'], inplace=True)
#
# # 'review' 열에서 한글 문자, 자음, 모음, 공백 문자 이외의 모든 문자 제거
# excel_data['review'] = excel_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
#
# # 전처리 후 처음 몇 개 행 확인
# print(excel_data.head())
#
# # 새로운 엑셀 파일로 저장
# excel_data.to_excel('전처리결과(3)_제거.xlsx', index=False)

# # ================================================

# import random
# import pandas as pd
#
# # 엑셀 파일 불러오기
# data = pd.read_excel('전처리결과(2)label.xlsx')
#
# # 데이터프레임의 행 수를 가져옵니다
# num_rows = len(data)
#
# # 2만개의 무작위 행 인덱스를 생성
# random_indices = random.sample(range(num_rows), 20000)
#
# # 무작위로 선택된 행을 추출
# random_sample = data.iloc[random_indices]
#
# # 인덱스를 오름차순으로 정렬
# random_sample = random_sample.sort_index(ascending=True)
# print(random_sample)
#
# # 데이터를 새로운 엑셀 파일로 저장
# random_sample.to_excel('전처리결과(2)_20000.xlsx', index=False)

# ================================================

import pandas as pd
import random

# 엑셀 파일 불러오기
data = pd.read_excel('전처리결과(3)_이진분류.xlsx')

# 'label' 열에서 0과 1로 이루어진 데이터 분리
label_0 = data[data['rating'] == 0]
label_1 = data[data['rating'] == 1]

# 0과 1 각각 2000개씩 추출
label_0_sample = label_0.sample(n=2000, random_state=1)
label_1_sample = label_1.sample(n=2000, random_state=1)

# 추출된 데이터를 하나의 데이터프레임으로 결합
final_sample = pd.concat([label_0_sample, label_1_sample])

# 결과를 셔플하여 순서를 랜덤화
final_sample = final_sample.sample(frac=1, random_state=1).reset_index(drop=True)

# 결과를 새로운 엑셀 파일로 저장
final_sample.to_excel('전처리결과(3)_per2000.xlsx', index=False)

