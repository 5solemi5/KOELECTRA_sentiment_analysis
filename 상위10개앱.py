import pandas as pd

# 엑셀 파일을 pandas DataFrame으로 읽습니다.
df = pd.read_excel('전처리결과(3)_문장길이.xlsx')

# 건강앱 이름별로 개수를 세고, 상위 3개만 선택합니다.
top_3_apps = df['app'].value_counts().nlargest(3).index

# 건강앱 이름이 상위 3개에 속하는 행만 선택하여 새로운 DataFrame을 생성합니다.
df_new = df[df['app'].isin(top_3_apps)]

# 결과를 새로운 엑셀 파일로 저장합니다.
df_new.to_excel('전처리결과(3)_상위3개앱.xlsx', index=False)
