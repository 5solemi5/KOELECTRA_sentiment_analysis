<div align=center><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/1a6d2fa0-ae24-4426-aa26-bfcba7ba4690" width="800">

 
# :hospital:  건강 관리앱 리뷰 감성분석 :iphone:
  
**KOELECTRA를 활용한 긍부정 예측 딥러닝 프로젝트**
  
건강 관리앱 리뷰를 이진 분류 문제로 정의하여 KOELECTRA 모델을 훈련시킨다. 
  
<h2>:heavy_check_mark:Tech Stack</h2>
<img src="https://img.shields.io/badge/PyTorch-E34F26?style=flat-square&logo=PyTorch&logoColor=white"/></a> 
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/></a>

</div>

# 1. 개요
## 1.1 문제 정의

디지털헬스케어란 통상 ICT 등 디지털 기술을 활용해 질병을 진단·치료하고 건강의 유지·증진을 목적으로 하는 일련의 활동과 수단을 의미한다.
  
<div align=center>
  
![디지털 헬스분야 매출 현황](https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/681f4793-5b8c-49ea-a446-5d9b3385e354)
 
[[자료: 디지털 헬스분야 매출 현황](https://www.dailypharm.com/Users/News/NewsView.html?ID=298670)]

  </div>

글로벌 디지털헬스케어 산업은 질병의 사후적 진단·치료에서 선제적 예방·관리로 의료 패러다임이 변화함에 따라 최근 6년간(’14~‘20) 연평균 39% 성장했고 향후(’20~‘27) 연평균 18.8%의 높은 성장세를 이어나갈 것으로 전망된다. 2021년 기준 매출은 1조 8227억원으로 전년 대비 34.6% 성장한 것으로 나타났다. 의료용기기 매출이 9731억원(53.4%)으로 가장 높았고 건강관리 기기 2546억원, 디지털 건강관리 플랫폼2250억원으로 뒤를 이었다. [<sup>[1]</sup>](https://www.dailypharm.com/Users/News/NewsView.html?ID=298670)

<div align=center><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/90f62896-0d86-4892-8ed3-0959807994fb" width="700">

[자료: 건강 관리앱 아이콘들]
  
 </div>

 위의 내용을 통해 현재 디지털헬스케어 시장은 전 세계적으로 급성장하고 있다는 것을 알 수 있다. 그리고 이에 따라 다양한 건강 관리 앱들이 출시되고 있다. 그러나 국내 디지털 헬스 산업은 기술적인 발전에도 불구하고 법 제도적인 문제로 상용화에 어려움을 겪고 있는 실정이다.  [<sup>[2]</sup>](https://www.bioin.or.kr/board.do?num=309481&bid=industry&cmd=view) 이런 상황에서 사용자들의 의견과 감성은 앱 개발 및 개선, 그리고 관련 규제 개선에 중요한 인사이트를 제공할 수 있다. 사용자들의 만족도는 앱마다 크게 차이가 나며, 이는 앱 리뷰를 통해 확인할 수 있다. 따라서 본 프로젝트에서는 건강 관리 앱의 리뷰 데이터를 활용하여 감성 분석을 수행하려고 한다.

본 프로젝트의 목표는 주요 건강 관리 앱들의 리뷰 데이터를 분석하여 긍부정 리뷰 비율, 사용자 선호도 등을 파악하는 것이다. 이를 통해 사용자 만족도 향상 방안을 도출하며, 필요한 법 제도 개선 사항 등을 제안하는 것이 가능할 것이다. 그리고 이와 같은 문제 정의를 설정함으로써 다양한 디지털 헬스케어 기술의 상용화를 촉진하는 방향으로 활용되고 국내 디지털 헬스 산업 발전에 중요한 인사이트를 제공하는 것을 기대한다.


## 1.2 데이터 및 모델 개요

- 데이터:
  
  한국 구글 플레이 스토어 내 건강관리 앱 424개의 리뷰를 수집한 원시 데이터를 목적에 맞게 전처리하여 사전 학습 언어 모델의 재학습(fine-tuning)을 위한 입력 데이터를 생성한다. 

- KOELECTRA 모델:


# 2. 데이터
## 2.1 데이터 소스

- 원시 데이터 출처: (https://github.com/park-gb/mHealthApp-review-textmining)

## 2.2 탐색적 데이터 분석

- 원시 데이터 정보:

<div align=center>

|Index|app|review|rating|
|-|-|-|-|
|1|다리 근육 운동 – 4주 프로그램|다른 P4P어플과 연동 하면 기존에 있던 스케쥴이 싹 사라짐|1|
|2|다리 근육 운동 – 4주 프로그램|굿|5|
|3|다리 근육 운동 – 4주 프로그램|최고입니디|5|
|4|다리 근육 운동 – 4주 프로그램|아무곳에서나 보고 억지로라도 운동할수 있어서 너무 ...|5|
|...|...|...|...|
|540076|돈버는어플 - 캐시런|진짜 최고에오ㅠㅠㅠㅠㅠㅠㅠㅠㅠ친구들한테 알려줬더니 ...|5|

</div>

<div align=center>

[자료: 원시 데이터의 형태]

</div>

454가지의 app, 540,076여개의 review가 있고 rating은 부정에서 긍정을 0~5사이에서 점수를 매겼다. 

일부, 영어와 이모티콘이 포함된 review들이 존재한다.


- 원시 데이터 분석:

<div><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/b212a979-292e-4573-a548-4c6e9fb3b955" width="480"></div>

## 2.3 데이터 전처리

- 입력 데이터의 전처리 과정:

(1) Review에서 한글이 아닌 리뷰, 중복, 결측치를 제거한다.

프로젝트의 목적이 한글 텍스트에 대한 긍부정 예측이기 때문에 한글이 아닌 리뷰는 불필요한 정보가 된다. 이런 정보는 모델의 성능을 떨어뜨릴 수 있다. 그리고 같은 데이터를 반복적으로 학습하면 모델이 편향되어, 실제의 다양한 상황을 정확히 반영하지 못할 수 있다. 결측치는 데이터에 빈 공간을 만들어, 모델이 제대로 학습하지 못하게 만든다.

```
import pandas as pd

# 엑셀 파일 불러오기
excel_data = pd.read_excel('dataset_raw.xlsx')

# 중복 제거
excel_data.drop_duplicates(subset=['review'], inplace=True)

# 결측치가 있는 행 제거
excel_data.dropna(subset=['review'], inplace=True)

# 'review' 열에서 한글 문자, 자음, 모음, 공백 문자 이외의 모든 문자 제거
excel_data['review'] = excel_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

# 전처리 후 처음 몇 개 행 확인
print(excel_data.head())

# 새로운 엑셀 파일로 저장
excel_data.to_excel('전처리결과(3)_제거.xlsx', index=False)
```

결과:

|app 개수|review 개수|
|-|-|
|412개|287,970개|

(2) Review의 길이가 너무 길거나 짧은 문장들은 딥러닝 학습에 무의미한 데이터들이므로 제거한다. 문장 길이가 5~500인 Reviews만 추출한다

```
import pandas as pd

# 엑셀 파일 읽기
df = pd.read_excel("전처리결과(3)_제거.xlsx", engine='openpyxl')

# 각 리뷰의 길이 계산하고 데이터프레임 필터링
df['length'] = df['review'].str.len()
filtered_df = df[(df['length'] >= 5) & (df['length'] <= 500)]

# 필터링된 데이터를 새로운 엑셀 파일로 저장
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.to_excel("전처리결과(3)_문장길이.xlsx", index=False)
```

결과:

|app 개수|review 개수|
|-|-|
|412개|278,426개|


![길게최종](https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/ad1bb4d0-00b0-4c1a-a203-b216ef46c27d)


(3) 리뷰의 개수가 충분해서 학습 및 분석의 의미가 있는 앱들을 남긴다. 건강 관리 앱 412개 중에서 리뷰의 수가 상위 3위 안에 드는는 앱들만 남긴다.

```
import pandas as pd

# 엑셀 파일을 pandas DataFrame으로 읽습니다.
df = pd.read_excel('전처리결과(3)_문장길이.xlsx')

# 건강앱 이름별로 개수를 세고, 상위 3개만 선택합니다.
top_3_apps = df['app'].value_counts().nlargest(3).index

# 건강앱 이름이 상위 3개에 속하는 행만 선택하여 새로운 DataFrame을 생성합니다.
df_new = df[df['app'].isin(top_3_apps)]

# 결과를 새로운 엑셀 파일로 저장합니다.
df_new.to_excel('전처리결과(3)_상위3개앱.xlsx', index=False)
```

결과:

|app 개수|review 개수|
|-|-|
|3개|110,410개|


(5) rating 5, 4은 긍정(1), 2, 1, 0은 부정(0)으로 이진분류하고 rating 3의 review는 긍부정이 뚜렷하지 않다고 판단하여 제거했다.

결과:

평균값보다 높은 임계값을 기준으로 이진 분류했음에도 불구하고 긍정(1)에 데이터가 치우쳐 있다. 클래스 불균형은 모델의 학습에 부정적인 영향을 미칠 수 있다. 

|긍정(1)|부정(0)|
|-|-|
|14,046개|50,202개|

<div><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/32f8869a-a1f2-4636-b72e-6b1a0ae97637" width="570"></div>



- 학습에 활용할 데이터의 양

학습 정확도를 높이기 위해 긍정(1), 부정(0)에서 임의로 각각 1,000건씩 추출하여 2,000건을 추출한다. 앞서 가공한 분석 데이터에서 발생하는 클래스 불균형을 해결한다.

<div align=center>

|Index|app|review|rating|label|
|-|-|-|-|-|
|1|눔(Noom) - 건강관리...|눔 코치를 결재하고 나서 두달되갈때쯤 또다시 ...|3|0|
|2|Nike Run Club|비밀번호 풀리면 다시 들어갈 방법이  ...|1|0|
|3|캐시슬라이드 스텝업 ...|아주더러운앱오류심하고오작동매우심하며 ...|1|0|
|4|Nike Run Club|달리기앱으로는  최고인듯|5|1|
|...|...|...|...|...|
|4001|눔(Noom) - 건강관리...|좋긴 한데 업데이트 이후 시간하고..|3|0|

</div>

<div align=center>

[자료: 학습 데이터의 구성]

</div>

<div><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/60fd406b-9012-47ef-91d1-19cb507ff49e" width="570"></div>
  
- 학습과 검증 데이터셋 분리

|학습|검증|
|-|-|
|3,200개|800개|


# 3. 재학습 결과
## 3.1 개발 환경
 - pycharm, python, torch, pandas, ...
 - 
## 3.2 KOELECTRA fine-tuning
## 3.3 학습 결과 그래프

# 4. 배운점

# Reference

[1]https://www.dailypharm.com/Users/News/NewsView.html?ID=298670

[2]https://www.bioin.or.kr/board.do?num=309481&bid=industry&cmd=view
