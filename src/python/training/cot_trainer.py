"""Long CoT (Chain of Thought) trainer with plan alignment"""

import torch
from typing import List, Tuple

class CoTTrainer:
    def __init__(self, model, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train_step(self, prompt: str, reasoning_steps: List[str], answer: str) -> float:
        """Train on CoT with plan alignment"""
        self.optimizer.zero_grad()

        # Concatenate reasoning steps
        full_sequence = prompt + " " + " ".join(reasoning_steps) + " " + answer
        input_ids = self._tokenize(full_sequence)
        target_ids = input_ids.clone()

        # Forward pass
        logits = self.model(input_ids)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text (simplified)"""
        return torch.randint(0, 50000, (1, len(text.split())))
