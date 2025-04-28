import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file, header=None)
        self.labels = data.iloc[:, 0].values.astype('float32')
        self.features = data.iloc[:, 1:].values.astype('float32')
        if self.features.shape[1] != 10:
            raise ValueError(f"Expected 10 features, got {self.features.shape[1]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).view(2, 5)
        y = torch.tensor(self.labels[idx]).unsqueeze(0)
        return x, y


class VGG1D(nn.Module):
    def __init__(self, dropout_rate=0.5):
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

            # block3 (with pooling)
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=3, padding=1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),

            # adaptive pooling
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1)
        )  # logits output

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits.view(-1), y_batch.view(-1))
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
            logits = model(x_batch)
            loss = criterion(logits.view(-1), y_batch.view(-1))
            total_loss += loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds.view(-1) == y_batch.view(-1)).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100 * correct / len(loader.dataset)
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train 1D-CNN VGG1D model')
    parser.add_argument(
        '-tr', '--train-csv', dest='train_csv',
        type=str, required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '-te', '--test-csv', dest='test_csv',
        type=str, required=True,
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size',
        type=int, default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr', dest='lr',
        type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs', dest='epochs',
        type=int, default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--patience', dest='patience',
        type=int, default=5,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--dropout', dest='dropout',
        type=float, default=0.5,
        help='Dropout rate for classifier'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = CustomDataset(args.train_csv)
    test_ds  = CustomDataset(args.test_csv)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = VGG1D(dropout_rate=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_loop(model, test_loader, criterion, device)
        _, train_acc = eval_loop(model, train_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered. Training halted.")
                break

    # Visualization
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'],   label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()