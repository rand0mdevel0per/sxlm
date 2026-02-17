"""Supervised Fine-Tuning (SFT) trainer"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from tqdm import tqdm

class SFTTrainer:
    def __init__(self, model, learning_rate: float = 1e-4, device: str = "cuda"):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total
        }
