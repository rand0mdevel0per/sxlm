"""PTE Executor: Plan-Think-Execute flow orchestrator"""

import torch
import torch.nn as nn
from .planner_port import PlannerPort

class PTEExecutor(nn.Module):
    def __init__(self, model, dim: int):
        super().__init__()
        self.model = model
        self.planner = PlannerPort(dim)

    def plan(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan phase: Generate implicit instruction sequence"""
        z, effort = self.planner(context)
        return z, effort

    def think(self, input_ids: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Think phase: Second read with biased attention guided by plan"""
        # Second read with plan-induced biased attention
        # z guides which parts of context to focus on
        hidden_states = self.model(input_ids, attention_bias=z)
        return hidden_states

    def execute(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute phase: Generate final answer or tool calls"""
        output = self.model.lm_head(hidden_states)
        return output

    def forward(self, input_ids: torch.Tensor, full_context: torch.Tensor):
        """Full PTE flow"""
        # Phase 1: Plan
        z, effort = self.plan(full_context)

        # Phase 2: Think (second read with biased attention)
        hidden_states = self.think(input_ids, z)

        # Phase 3: Execute
        output = self.execute(hidden_states)

        return output, z, effort
