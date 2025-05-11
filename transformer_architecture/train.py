# File: train.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import TransformerDecoder
from tqdm import tqdm
import os

block_size = 128
batch_size = 64
embed_dim = 128
epochs = 20
lr = 3e-4

with open("data/train.txt", "r") as f: train_data = f.read()
with open("data/val.txt", "r") as f: val_data = f.read()
chars = sorted(list(set(train_data + val_data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

class CharDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(encode(data), dtype=torch.long)

    def __len__(self):
        return len(self.data) - block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+block_size]
        y = self.data[idx+1:idx+block_size+1]
        return x, y

train_loader = DataLoader(CharDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CharDataset(val_data), batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TransformerDecoder(vocab_size, block_size, embed_dim=embed_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

def train_model():
    best_loss = float('inf')
    patience, counter = 3, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    train_model()