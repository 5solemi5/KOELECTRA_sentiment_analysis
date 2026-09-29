import pandas as pd
from pykospacing import spacing
from concurrent.futures import ThreadPoolExecutor

# 엑셀 파일 읽기
df = pd.read_excel('dataset_raw (1).xlsx')

def correct_spacing(text):
    try:
        return spacing(text)
    except Exception as e:
        print(f"Error: {e}, on text: {text}")
        return text

def apply_spacing(row):
    row['review'] = correct_spacing(row['review'])
    print(f"Completed for index: {row.name}")
    return row

# ThreadPoolExecutor 생성 (스레드 개수는 컴퓨터 성능에 따라 조절)
executor = ThreadPoolExecutor(max_workers=10)

# B열 데이터에 대해 PyKoSpacing 적용
with executor as e:
    futures = [e.submit(apply_spacing, row) for _, row in df.iterrows()]

# 결과 확인 및 저장
df.to_excel('corrected_dataset.xlsx', index=False)
