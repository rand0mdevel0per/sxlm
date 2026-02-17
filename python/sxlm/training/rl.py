"""RL training with eligibility traces"""

import torch
import torch.nn as nn
from typing import Dict, Callable

class EligibilityTrace:
    def __init__(self, model: nn.Module, decay: float = 0.9):
        self.model = model
        self.decay = decay
        self.traces = {name: torch.zeros_like(param)
                      for name, param in model.named_parameters()}

    def update_traces(self, loss: torch.Tensor):
        """Update eligibility traces with current gradients"""
        grads = torch.autograd.grad(loss, self.model.parameters(),
                                    retain_graph=True, create_graph=False)
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if grad is not None:
                self.traces[name] = self.decay * self.traces[name] + grad

    def apply_reward(self, reward: float):
        """Apply reward to parameters using traces"""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = reward * self.traces[name]
            else:
                param.grad += reward * self.traces[name]

class RLTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4,
                 trace_decay: float = 0.9, device: str = "cuda"):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.el_trace = EligibilityTrace(model, decay=trace_decay)

    def train_step(self, state: torch.Tensor, action: torch.Tensor,
                   reward: float, next_state: torch.Tensor) -> float:
        """Single RL training step with eligibility traces"""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(state)
        loss = nn.functional.cross_entropy(logits, action)

        # Update eligibility traces
        self.el_trace.update_traces(loss)

        # Apply reward
        self.el_trace.apply_reward(reward)

        # Optimizer step
        self.optimizer.step()

        return loss.item()
