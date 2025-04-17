import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1) Custom Dataset (경고 해결)
# =============================================================================
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        self.labels   = data.iloc[:,0].values.astype('float32')
        self.features = data.iloc[:,1:].values.astype('float32')  # shape (N,10)
        if self.features.shape[1] != 10:
            raise ValueError(f"특징 열 개수 오류: {self.features.shape[1]} != 10")
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.features[idx]          # numpy array (10,)
        # 한 번에 변환 & 리쉐이프
        x = torch.from_numpy(sample).view(2,5)  # shape (2,5)
        y = torch.tensor(self.labels[idx]).unsqueeze(0)  # shape (1,)
        return x, y

# =============================================================================
# 2) VGG19‑1D 정의 (블록3 풀링 제거)
# =============================================================================
class VGG1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # block1
            nn.Conv1d(2,  64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2,2), 

            # block2
            nn.Conv1d(64,128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128,128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2,2),

            # block3 — 풀링 제거!
            nn.Conv1d(128,256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256,256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256,256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256,512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            # nn.MaxPool1d(2,2),  #← 삭제

            # block4 & block5 (기존 그대로)
            # nn.Conv1d(256,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),

            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            # nn.Conv1d(512,512, kernel_size=3, padding=1), nn.ReLU(inplace=True),

            # 최종 길이 1로 고정
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                     
            nn.Linear(512,4096), nn.ReLU(inplace=True),# nn.Dropout(),
            # nn.Linear(4096,4096), nn.ReLU(inplace=True),# nn.Dropout(),
            nn.Linear(4096,   1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============================================================================
# 3) 학습/평가 루프는 그대로 사용
# =============================================================================
def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        # print(y_pred, y_batch)
        loss   = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_batch.size(0)
    return total_loss / len(loader.dataset)

def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            total_loss += criterion(y_pred, y_batch).item() * x_batch.size(0)
            correct += ((y_pred >= 0.5).float() == y_batch).sum().item()
    return total_loss / len(loader.dataset), 100 * correct / len(loader.dataset)

# =============================================================================
# 4) 실행부
# =============================================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    train_ds = CustomDataset(r'C:\Users\admin\Desktop\1D-CNN\TrainDataset_5s, TestDataset_5s\Traindataset_Augmentation_Liftupdown24,Driving046_5s.csv')
    test_ds  = CustomDataset(r'C:\Users\admin\Desktop\1D-CNN\TrainDataset_5s, TestDataset_5s\TestDataset_Liftupdown3,Driving28_5s.csv')
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model     = VGG1D().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for ep in range(1, 10000):
        tl = train_loop(model, train_loader, criterion, optimizer, device)
        vl, acc = eval_loop(model, test_loader, criterion, device)
        print(f"Epoch {ep:02d} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | Acc: {acc:.2f}%")
