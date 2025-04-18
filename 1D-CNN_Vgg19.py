import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Custom Dataset 정의
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        self.labels   = data.iloc[:, 0].values.astype('float32')
        self.features = data.iloc[:, 1:].values.astype('float32')  # (N,10)
        if self.features.shape[1] != 10:
            raise ValueError(f"Expected 10 features, got {self.features.shape[1]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).view(2, 5)  # (2,5)
        y = torch.tensor(self.labels[idx]).unsqueeze(0)     # (1,)
        return x, y

# 2) VGG1D 네트워크 정의 (불필요한 블록 제거)
class VGG1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # block1
            nn.Conv1d(2, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),

            # block2
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),

            # block3 (풀링 제거)
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),

            # Adaptive Pooling으로 길이 고정
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3) 학습/평가 루프 정의

def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)


def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            total_loss += criterion(y_pred, y_batch).item() * x_batch.size(0)
            correct += ((y_pred >= 0.5).float() == y_batch).sum().item()
    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)

# 4) 메인 실행부 및 통합 시각화
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 데이터셋 경로 설정
    train_ds = CustomDataset(r'C:\Users\admin\Desktop\1D-CNN\TrainDataset_5s, TestDataset_5s\Traindataset_Augmentation_Liftupdown24,Driving046_5s.csv')
    test_ds  = CustomDataset(r'C:\Users\admin\Desktop\1D-CNN\TrainDataset_5s, TestDataset_5s\TestDataset_Liftupdown3,Driving28_5s.csv')
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 모델, 손실 함수, 옵티마이저
    model     = VGG1D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # 학습 및 평가
    for ep in range(1, epochs+1):
        tl = train_loop(model, train_loader, criterion, optimizer, device)
        vl, test_acc = eval_loop(model, test_loader, criterion, device)
        _, train_acc = eval_loop(model, train_loader, criterion, device)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(train_acc)
        val_accs.append(test_acc)
        print(f"Epoch {ep:03d} | Train Loss: {tl:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {vl:.4f} | Test Acc: {test_acc:.2f}%")

    # 5) Loss와 Accuracy 통합 시각화
    epochs_range = range(1, epochs+1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(epochs_range, train_losses, label='Train Loss')
    ax1.plot(epochs_range, val_losses,   label='Test Loss')
    ax2.plot(epochs_range, train_accs,    label='Train Accuracy')
    ax2.plot(epochs_range, val_accs,      label='Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title('Loss and Accuracy over Epochs')
    plt.show()
