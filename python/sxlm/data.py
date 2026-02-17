"""Data loading utilities for SXLM training"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class TextDataset(Dataset):
    def __init__(self, texts: List[str], max_len: int = 512):
        self.texts = texts
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Simple character-level tokenization
        tokens = [ord(c) % 50257 for c in text[:self.max_len]]

        if len(tokens) < 2:
            tokens = [0, 0]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}

def collate_fn(batch):
    """Collate function with padding"""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[i, :length] = item["input_ids"]
        labels[i, :length] = item["labels"]

    return {"input_ids": input_ids, "labels": labels}
