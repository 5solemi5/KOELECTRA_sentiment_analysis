# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # 엑셀 파일 불러오기
# # file_path = 'dataset_raw (1).xlsx'
# # data = pd.read_excel('dataset_raw (1).xlsx')
# #
# # # rating 별 review 개수 계산
# # rating_counts = data['rating'].value_counts().sort_index()
# #
# # # Rating 별 Review 개수 시각화 (x축 0 제거)
# # plt.figure(figsize=(8, 5))
# #
# # # 막대의 높이에 따라 색상을 그라데이션으로 변경
# # colors = sns.color_palette("YlGnBu", len(rating_counts[rating_counts.index != 0].values))
# # normalized_values = (rating_counts[rating_counts.index != 0].values - min(rating_counts)) / (max(rating_counts) - min(rating_counts))
# #
# # bars = plt.bar(rating_counts.index[rating_counts.index != 0], rating_counts[rating_counts.index != 0].values, alpha=0.8, color=colors)
# #
# # # 각 막대 중앙에 review 개수 숫자 표시
# # for bar, val, color in zip(bars, rating_counts[rating_counts.index != 0].values, colors):
# #     yval = bar.get_height()
# #     plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')
# #
# # # 그래프 디자인
# # plt.xlabel('review', fontsize=12)
# # plt.ylabel('Number of reviews', fontsize=12)
# # plt.title('Number of reviews', fontsize=14)
# # plt.grid(axis='y', linestyle='--', alpha=0.7)
# # plt.ylim(0, max(rating_counts) + 20)
# # plt.xticks(fontsize=10)
# # plt.yticks(fontsize=10)
# # plt.tight_layout()
# #
# # # 전체 테두리 없애기
# # for spine in plt.gca().spines.values():
# #     spine.set_visible(False)
# #
# # # 막대의 모서리를 둥글게 만듭니다.
# # for bar in bars:
# #     bar.set_edgecolor('grey')
# #     bar.set_linewidth(1)
# #     bar.set_alpha(0.8)
# #
# # plt.show()
# #
# #
# # ======================================================================

import matplotlib.pyplot as plt

# 데이터
positive_count = 74078  #50202 #2000 23547|74078
negative_count = 23547 #14046 #2000
labels = ['Positive', 'Negative']
sizes = [positive_count, negative_count]
colors = ['#FA9CC5', '#B2EBF2']  # 색상
explode = (0.1, 0)  # 파이 차트에서 분리 표현할 부분

# 파이 차트 생성
plt.figure(figsize=(8, 6))
_, texts, _ = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode, textprops={'color': 'white', 'fontsize': '27'}, shadow=True)

# 차트 타이틀
plt.title('Positive vs Negative Reviews')

# Positive 라벨의 글자 색상을 변경
texts[0].set_color('white')

plt.axis('equal')  # 파이 차트를 원형으로 유지
plt.show()

# # =====================================================================
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 엑셀 파일 불러오기
# file_path = 'dataset_raw (1).xlsx'
# data = pd.read_excel('dataset_raw (1).xlsx')
#
# # rating 별 review 개수 계산
# rating_counts = data['rating'].value_counts().sort_index()
#
# # Rating 별 Review 개수 시각화 (x축 0 제거)
# plt.figure(figsize=(8, 5))
#
# # 사용할 5가지 색상
# custom_colors = ['#4BAF4B', '#FFBB00', '#828282', '#FF8C00', '#003399']
#
# bars = plt.bar(rating_counts.index[rating_counts.index != 0], rating_counts[rating_counts.index != 0].values, color=custom_colors)
#
# # 각 막대 중앙에 review 개수 숫자 표시
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')
#
# # 그래프 디자인
# plt.xlabel('review', fontsize=12)
# plt.ylabel('Number of reviews', fontsize=12)
# plt.title('Number of reviews', fontsize=14)
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.ylim(0, max(rating_counts) + 20)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
#
# # 전체 테두리 없애기
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
#
# # 막대의 모서리를 둥글게 만듭니다.
# for bar in bars:
#     bar.set_edgecolor('grey')
#     bar.set_linewidth(1)
#     bar.set_alpha(0.8)
#
# plt.show()
#
# ===================================
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 엑셀 파일 불러오기
# file_path = 'dataset_raw.xlsx'
# data = pd.read_excel('dataset_raw.xlsx')
#
# # rating 별 review 개수 계산
# rating_counts = data['rating'].value_counts().sort_index()
#
# # Rating 별 Review 개수 시각화 (x축 0 제거)
# plt.figure(figsize=(5.5, 5))
#
# # 사용할 5가지 색상
# # custom_colors = ['#FA9CC5', '#B2EBF2', '#3C3C8C', '#FFC6C3', '#dda0dd']
# custom_colors = ['#dda0dd', '#3C3C8C', '#B2EBF2', '#FA9CC5', '#FFC6C3']
#
# bars = plt.bar(rating_counts.index[rating_counts.index != 0], rating_counts[rating_counts.index != 0].values, color=custom_colors)
#
# # 각 막대 중앙에 review 개수 숫자 표시
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', color='black', fontsize=9)
#
# # 그래프 디자인
# plt.xlabel('review', fontsize=12)
# plt.ylabel('Number of reviews', fontsize=12)
# plt.title('Number of reviews', fontsize=14)
#
# plt.grid(axis='y', linestyle='dotted', color='#E1F5FE') #alpha=0.7,
# plt.ylim(0, max(rating_counts) + 50000)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# plt.tight_layout()
#
#
# # # 전체 테두리 없애기
# # for spine in plt.gca().spines.values():
# #     spine.set_visible(False)
#
# # 막대의 모서리를 둥글게 만듭니다.
# for bar in bars:
#     bar.set_edgecolor('white') #B2EBF2
#     bar.set_linewidth(1.3)
#     bar.set_alpha(1.0)
#
#
# plt.show()
