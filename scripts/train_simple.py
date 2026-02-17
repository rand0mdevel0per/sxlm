"""Pure PyTorch SXLM training (no C++ dependencies)"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SimpleConfig:
    dim = 768
    num_heads = 12
    num_layers = 24
    max_seq_len = 2048
    learning_rate = 1e-4

class QuilaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(50257, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.dim))
        self.layers = nn.ModuleList([
            QuilaLayer(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, 50257, bias=False)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

class QuilaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(config.dim, config.num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.GELU(),
            nn.Linear(4 * config.dim, config.dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class TextDataset(Dataset):
    def __init__(self, texts, max_len=128):
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = [ord(c) % 50257 for c in text[:self.max_len]]
        if len(tokens) < 2:
            tokens = [0, 0]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[i, :length] = item["input_ids"]
        labels[i, :length] = item["labels"]
    return {"input_ids": input_ids, "labels": labels}

def train():
    config = SimpleConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = QuilaModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_texts = ["Hello world! This is SXLM training. " * 5] * 1000
    train_dataset = TextDataset(train_texts, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "E:/sxlm/quila_model.pt")
    print("Model saved!")

if __name__ == "__main__":
    train()
