<div align=center><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/45bbeb60-f509-49a5-8df7-a5010227358f" width="900">

 
# :hospital: :iphone: 건강 관리앱 리뷰 감성분석 :pill: :syringe: 
  
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

[자료: 원시 데이터의 형태]

454가지의 app, 540,076여개의 review가 있고 rating은 부정에서 긍정을 0~5사이에서 점수를 매겼다. 

일부, 영어와 이모티콘이 포함된 review들이 존재한다.

</div>

- 원시 데이터 분석:

<div align=center><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/7a1893a1-02ce-4517-9b56-648dcb8326e6">

[자료: Number of reviews by rating]

</div>

## 2.3 데이터 전처리

- 입력 데이터의 전처리 과정:


(1) 원시 데이터에서 건강 관리 앱 424개 중에서 리뷰의 수가 충분히 있는 앱들만 남긴다.

결과:

|app 개수|review 개수|rating 평균|
|-|-|-|
|44개|468,814개|4.355578|


(2) 그 중에서 개선의 여지가 있는 평균 평점의 앱들과 리뷰의 수가 많은 앱들만 남긴다.

원시 데이터의 rating의 평균이 4.355578이므로 rating의 평균이 4.1~4.4 사이에 있는 앱들과 리뷰의 수가 많은 앱들만 남긴다.

결과:

|app 개수|review 개수|rating 평균|
|-|-|-|
|9개|152,592개|4.370255321|

(3) Review에서 한글이 아닌 리뷰, 중복, 결측치를 제거한다.

결과:

|app 개수|review 개수|rating 평균|
|-|-|-|
|9개|64,248개|4.160658698|

<div><img src = "https://github.com/5solemi5/KOELECTRA_sentiment_analysis/assets/104000117/3de8afd7-21e0-4b2e-9477-edc1bbab9424" width="380"></div>

(4) Rating 5, 4은 긍정(1), 3, 2, 1, 0은 부정(0)으로 이진분류를 한다.

결과:

|긍정(1)|부정(0)|
|-|-|
|14,046개|50,202개|

평균값보다 높은 임계값을 기준으로 이진 분류했음에도 불구하고 긍정(1)에 데이터가 치우쳐 있다. 클래스 불균형은 모델의 학습에 부정적인 영향을 미칠 수 있는 다음과 같은 이유로 인해 문제가 될 수 있다. 


- 학습에 활용할 데이터의 양

학습 정확도를 높이기 위해 긍정(1), 부정(0)에서 임의로 각각 10,000건씩 추출하여 20,000건을 추출한다. 앞서 가공한 분석 데이터에서 발생하는 클래스 불균형을 해결한다.

|Index|app|review|rating|label|
|-|-|-|-|-|
|1|눔(Noom) - 건강관리...|눔 코치를 결재하고 나서 두달되갈때쯤 또다시 ...|3|0|
|2|Nike Run Club|비밀번호 풀리면 다시 들어갈 방법이  ...|1|0|
|3|캐시슬라이드 스텝업 ...|아주더러운앱오류심하고오작동매우심하며 ...|1|0|
|4|Nike Run Club|달리기앱으로는  최고인듯|5|1|
|...|...|...|...|...|
|4001|눔(Noom) - 건강관리...|좋긴 한데 업데이트 이후 시간하고..|3|0|

[자료: 학습 데이터의 구성]
  
- 학습과 검증 데이터셋 분리


# 3. 재학습 결과
## 3.1 개발 환경
 - pycharm, python, torch, pandas, ...
 - 
## 3.2 KOELECTRA fine-tuning
## 3.3 학습 결과 그래프

# 4. 배운점

# Reference
