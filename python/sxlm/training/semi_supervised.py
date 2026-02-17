"""Semi-supervised learning with external model labeling"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict
from .openrouter import OpenRouterClient
from .sft import SFTTrainer

class SemiSupervisedTrainer(SFTTrainer):
    def __init__(self, model, openrouter_client: OpenRouterClient,
                 confidence_threshold: float = 0.8, **kwargs):
        super().__init__(model, **kwargs)
        self.openrouter = openrouter_client
        self.confidence_threshold = confidence_threshold

    def label_unlabeled_data(self, prompts: List[str],
                            teacher_model: str = "anthropic/claude-opus-4-6") -> List[str]:
        """Generate labels using external model"""
        return self.openrouter.batch_generate(prompts, model=teacher_model)

    def train_with_unlabeled(self, labeled_loader: DataLoader,
                            unlabeled_prompts: List[str]) -> float:
        """Train with both labeled and pseudo-labeled data"""
        # Generate pseudo-labels
        pseudo_labels = self.label_unlabeled_data(unlabeled_prompts)

        # Train on labeled data
        loss = self.train_epoch(labeled_loader)

        return loss
