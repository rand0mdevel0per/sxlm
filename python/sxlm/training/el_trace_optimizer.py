"""el-trace: Eligibility trace for parameter-level credit assignment"""

import torch
import torch.nn as nn

class EligibilityTraceOptimizer:
    def __init__(self, model: nn.Module, decay: float = 0.9):
        self.model = model
        self.decay = decay
        self.traces = {}

        # Initialize traces for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.traces[name] = torch.zeros_like(param)

    def update_traces(self, loss: torch.Tensor):
        """Update eligibility traces with current gradients"""
        grads = torch.autograd.grad(loss, self.model.parameters(),
                                     retain_graph=True, create_graph=False)

        for (name, param), grad in zip(self.model.named_parameters(), grads):
            if param.requires_grad and grad is not None:
                # Decay old trace and add new gradient
                self.traces[name] = self.decay * self.traces[name] + grad.detach()

    def apply_reward(self, reward: float):
        """Apply reward signal to parameters via traces"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.traces:
                if param.grad is None:
                    param.grad = reward * self.traces[name]
                else:
                    param.grad += reward * self.traces[name]

    def apply_multi_reward(self, rewards: dict):
        """
        Apply multi-dimensional rewards

        Args:
            rewards: Dict with keys 'usefulness', 'conciseness', 'anti_hallucination'
        """
        r_use = rewards.get('usefulness', 0.0)
        r_concise = rewards.get('conciseness', 0.0)
        r_faith = rewards.get('anti_hallucination', 0.0)

        total_reward = r_use + r_concise + r_faith
        self.apply_reward(total_reward)
