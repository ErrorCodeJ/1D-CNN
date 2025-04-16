import pandas as pd
import numpy as np

# CSV 파일 경로 (사용자 환경에 맞게 수정)
csv_path = r'C:\Users\admin\Desktop\1D-CNN\Liftupdown2_process data.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_path)

# 슬라이딩 윈도우 함수 (window_size=5, step=1)
def create_overlapping_windows(data, window_size=5, step=1):
    """
    data: 1차원 numpy 배열
    window_size: 한 윈도우에 포함될 데이터 개수 (여기서는 5)
    step: 윈도우가 이동하는 간격 (여기서는 1, 즉 1초마다)
    
    반환: overlapping 윈도우를 리스트 형태로 반환
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)
    return windows

# PackVolt, Current 컬럼 값 추출 (1차원 배열)
packvolt_values = df['PackVolt'].values
current_values = df['Current'].values

# 각각 overlapping 윈도우 생성 (윈도우 크기는 5, 1초씩 이동)
packvolt_windows = create_overlapping_windows(packvolt_values, window_size=5, step=1)
current_windows = create_overlapping_windows(current_values, window_size=5, step=1)

# 생성된 윈도우의 개수 확인 (예: 29개의 행에서 29 - 5 + 1 = 25개의 윈도우)
print(f"총 증강된 PackVolt 데이터의 개수: {len(packvolt_windows)}")
print(f"총 증강된 Current 데이터의 개수: {len(current_windows)}\n")

# 모든 증강 데이터(윈도우) 출력 (터미널에 모든 값을 출력)
print("전체 증강된 PackVolt 데이터:")
for idx, window in enumerate(packvolt_windows, start=1):
    print(f"PackVolt 윈도우 {idx}: {window}")

print("\n전체 증강된 Current 데이터:")
for idx, window in enumerate(current_windows, start=1):
    print(f"Current 윈도우 {idx}: {window}")

# 증강된 데이터(윈도우)를 CSV 파일에 저장하기 위한 DataFrame 생성
# 각 행은 하나의 윈도우이며, 첫 번째 열은 PackVolt 값들, 두 번째 열은 Current 값들(각각 쉼표로 구분된 문자열)
augmented_data = []
for i in range(len(packvolt_windows)):
    packvolt_str = ", ".join(map(str, packvolt_windows[i]))
    current_str = ", ".join(map(str, current_windows[i]))
    augmented_data.append({"PackVolt_Window": packvolt_str, "Current_Window": current_str})

df_augmented = pd.DataFrame(augmented_data)

# 저장할 CSV 파일 경로 (원하는 경로로 수정)
output_csv_path = r'C:\Users\admin\Desktop\1D-CNN\augmented_data_windows.csv'
df_augmented.to_csv(output_csv_path, index=False)
print(f"\n증강된 데이터가 CSV 파일로 저장되었습니다: {output_csv_path}")
