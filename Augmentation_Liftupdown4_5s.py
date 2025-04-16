import pandas as pd
import numpy as np

# CSV 파일 경로 (사용자 환경에 맞게 수정)
csv_path = r'C:\Users\admin\Desktop\1D-CNN\Liftupdown4_chronologize.csv'

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

# PackVolt 윈도우를 별도의 열로 분리하여 DataFrame 생성
packvolt_columns = [f'PackVolt_{i+1}' for i in range(5)]
df_packvolt = pd.DataFrame(packvolt_windows, columns=packvolt_columns)

# Current 윈도우도 같은 방식으로 DataFrame 생성
current_columns = [f'Current_{i+1}' for i in range(5)]
df_current = pd.DataFrame(current_windows, columns=current_columns)

# 두 DataFrame을 열 방향으로 병합 (각 행이 하나의 윈도우에 해당)
df_augmented = pd.concat([df_packvolt, df_current], axis=1)

# A열을 추가하고 모든 값에 0을 입력 (첫번째 열 위치에 삽입)
df_augmented.insert(0, 'A', 0)

# 저장할 CSV 파일 경로 (원하는 경로로 수정)
output_csv_path = r'C:\Users\admin\Desktop\1D-CNN\augmentation_data_windows_Liftupdown4_5s.csv'

# header=False를 사용하여 CSV 파일에 열 이름을 저장하지 않음
df_augmented.to_csv(output_csv_path, index=False, header=False)
print(f"\n증강된 데이터가 헤더 없이 CSV 파일로 저장되었습니다: {output_csv_path}")
