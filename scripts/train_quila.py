"""Complete SXLM training script"""

import sys
sys.path.insert(0, 'E:/sxlm/build/python/Release')
sys.path.insert(0, 'E:/sxlm/python')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import _sintellix_native as native
from sxlm.model import QuilaModel
from sxlm.data import TextDataset, collate_fn
from tqdm import tqdm

def train():
    # Load config
    config = native.QuilaConfig.load("E:/sxlm/config.toml")
    print(f"Config: dim={config.dim}, layers={config.num_layers}, heads={config.num_heads}")

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = QuilaModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset
    train_texts = ["Hello world! " * 10] * 1000
    train_dataset = TextDataset(train_texts, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
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

    # Save model
    torch.save(model.state_dict(), "E:/sxlm/quila_model.pt")
    print("Model saved to quila_model.pt")

if __name__ == "__main__":
    train()
