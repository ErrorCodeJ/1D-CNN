import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Custom Dataset 정의
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)  # CSV 파일 읽기
        self.labels   = data.iloc[:, 0].values.astype('float32')  # 레이블 (첫 번째 열)
        self.features = data.iloc[:, 1:].values.astype('float32')  # 특징 (두 번째 열부터 마지막 열까지)
        if self.features.shape[1] != 10:
            raise ValueError(f"Expected 10 features, got {self.features.shape[1]}")  # 특징의 수가 10이 아니면 오류 발생

    def __len__(self):
        return len(self.labels)  # 데이터셋의 크기 반환

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).view(2, 5)  # (2, 5) 형태로 변환
        y = torch.tensor(self.labels[idx]).unsqueeze(0)     # (1,) 형태로 변환
        return x, y

# 2) VGG1D 네트워크 정의 (불필요한 블록 제거)
class VGG1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # block1: 첫 번째 블록
            nn.Conv1d(2, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # MaxPooling

            # block2: 두 번째 블록
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # MaxPooling

            # block3: 세 번째 블록 (풀링 제거)
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),

            # Adaptive Pooling으로 길이 고정
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 텐서 평탄화
            nn.Linear(512, 4096), nn.ReLU(inplace=True),  # 완전 연결층 1
            nn.Linear(4096, 1), nn.Sigmoid()  # 출력층 (이진 분류)
        )

    def forward(self, x):
        x = self.features(x)  # 특성 추출
        x = self.classifier(x)  # 분류
        return x

# 3) 학습/평가 루프 정의
# 학습 루프
def train_loop(model, loader, criterion, optimizer, device):
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # 데이터를 GPU로 이동
        optimizer.zero_grad()  # 기울기 초기화
        y_pred = model(x_batch)  # 예측
        loss = criterion(y_pred, y_batch)  # 손실 계산
        loss.backward()  # 기울기 계산
        optimizer.step()  # 파라미터 업데이트
        total_loss += loss.item() * x_batch.size(0)  # 손실 누적
    return total_loss / len(loader.dataset)  # 평균 손실 반환

# 평가 루프
def eval_loop(model, loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss, correct = 0, 0
    with torch.no_grad():  # 평가 시에는 기울기 계산을 하지 않음
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # 데이터를 GPU로 이동
            y_pred = model(x_batch)  # 예측
            total_loss += criterion(y_pred, y_batch).item() * x_batch.size(0)  # 손실 계산
            correct += ((y_pred >= 0.5).float() == y_batch).sum().item()  # 정확도 계산
    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)  # 평균 손실과 정확도 반환

# 4) 메인 실행부 및 통합 시각화
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부 확인
    print("Device:", device)

    # 데이터셋 경로 설정
    train_ds = CustomDataset(r'C:\Users\admin\Desktop\1D-CNN\TrainDataset_5s, TestDataset_5s\Traindataset_Augmentation_Liftupdown24,Driving046_5s.csv')
    test_ds  = CustomDataset(r'C:\Users\admin\Desktop\1D-CNN\TrainDataset_5s, TestDataset_5s\TestDataset_Liftupdown3,Driving28_5s.csv')
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # 학습 데이터 로더
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)  # 테스트 데이터 로더

    # 모델, 손실 함수, 옵티마이저 설정
    model     = VGG1D().to(device)  # 모델을 디바이스에 할당
    criterion = nn.BCELoss()  # 이진 크로스 엔트로피 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam 옵티마이저

    epochs = 20  # 학습할 에폭 수
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # 손실과 정확도를 기록할 리스트

    # 학습 및 평가 루프
    for ep in range(1, epochs+1):
        tl = train_loop(model, train_loader, criterion, optimizer, device)  # 학습 루프
        vl, test_acc = eval_loop(model, test_loader, criterion, device)  # 평가 루프
        _, train_acc = eval_loop(model, train_loader, criterion, device)  # 학습 데이터 정확도
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(train_acc)
        val_accs.append(test_acc)
        print(f"Epoch {ep:03d} | Train Loss: {tl:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {vl:.4f} | Test Acc: {test_acc:.2f}%")

    # 5) Loss와 Accuracy 통합 시각화 (색상 지정)
    epochs_range = range(1, epochs+1)
    fig, ax1 = plt.subplots()  # 첫 번째 축 생성
    ax2 = ax1.twinx()  # 두 번째 축 생성
    ax1.plot(epochs_range, train_losses, label='Train Loss', color='tab:blue')  # 학습 손실
    ax1.plot(epochs_range, val_losses,   label='Test Loss',  color='tab:orange')  # 테스트 손실
    ax2.plot(epochs_range, train_accs,    label='Train Accuracy', color='tab:green')  # 학습 정확도
    ax2.plot(epochs_range, val_accs,      label='Test Accuracy',  color='tab:red')  # 테스트 정확도
    ax1.set_xlabel('Epoch')  # X축 레이블
    ax1.set_ylabel('Loss')   # Y축 (손실)
    ax2.set_ylabel('Accuracy (%)')  # Y축 (정확도)
    lines1, labels1 = ax1.get_legend_handles_labels()  # 첫 번째 축의 레전드
    lines2, labels2 = ax2.get_legend_handles_labels()  # 두 번째 축의 레전드
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')  # 레전드 합치기
    plt.title('Loss and Accuracy over Epochs')  # 그래프 제목
    plt.show()  # 그래프 출력
