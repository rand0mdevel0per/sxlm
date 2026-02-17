"""SFT training script"""

import sys
sys.path.insert(0, 'E:/sxlm/python')

import torch
from torch.utils.data import Dataset, DataLoader
from sxlm.training import SFTTrainer
import _sintellix_native as sxlm

class TextDataset(Dataset):
    def __init__(self, texts, max_len=512):
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Placeholder tokenization
        tokens = [ord(c) % 256 for c in self.texts[idx][:self.max_len]]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

def main():
    # Load config
    config = sxlm.QuilaConfig.load("config.toml")

    # Create dummy model (replace with actual model)
    model = torch.nn.Linear(config.dim, 256).cuda()

    # Create dataset
    train_texts = ["Hello world"] * 100
    train_dataset = TextDataset(train_texts)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Train
    trainer = SFTTrainer(model, learning_rate=config.learning_rate)

    for epoch in range(10):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

if __name__ == "__main__":
    main()
