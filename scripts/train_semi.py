"""Semi-supervised training script"""

import sys
sys.path.insert(0, 'E:/sxlm/python')

import torch
from torch.utils.data import DataLoader
from sxlm.training import SemiSupervisedTrainer, OpenRouterClient
from train_sft import TextDataset
import _sintellix_native as sxlm
import os

def main():
    # Load config
    config = sxlm.QuilaConfig.load("config.toml")

    # Create model
    model = torch.nn.Linear(config.dim, 256).cuda()

    # OpenRouter client
    api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter = OpenRouterClient(api_key)

    # Create datasets
    labeled_texts = ["Hello world"] * 50
    unlabeled_prompts = ["Generate text"] * 50

    labeled_dataset = TextDataset(labeled_texts)
    labeled_loader = DataLoader(labeled_dataset, batch_size=8, shuffle=True)

    # Train
    trainer = SemiSupervisedTrainer(model, openrouter, learning_rate=config.learning_rate)

    for epoch in range(10):
        loss = trainer.train_with_unlabeled(labeled_loader, unlabeled_prompts)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

if __name__ == "__main__":
    main()
