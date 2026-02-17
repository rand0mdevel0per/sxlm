#!/usr/bin/env python3
import os
import sys
import toml
import torch
import deepspeed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from sxlm.training.sft import SFTTrainer
from sxlm.training.semi_supervised import SemiSupervisedTrainer
from sxlm.training.challenger import ChallengerAnswerGenerator

def load_config(config_path="config.toml"):
    with open(config_path) as f:
        return toml.load(f)

def main():
    config = load_config()

    # Initialize model (placeholder - will use actual SXLM model)
    model = torch.nn.TransformerDecoder(
        torch.nn.TransformerDecoderLayer(
            d_model=config["model"]["dim"],
            nhead=config["model"]["num_heads"]
        ),
        num_layers=config["model"]["num_layers"]
    )

    # DeepSpeed initialization
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config="deepspeed_config.json"
    )

    # Phase 1: SFT
    print("=== Phase 1: Supervised Fine-Tuning ===")
    sft_trainer = SFTTrainer(model_engine, optimizer, config)

    for step in range(config["training"]["sft_steps"]):
        loss = sft_trainer.train_step()

        if step % config["training"]["logging_steps"] == 0:
            print(f"SFT Step {step}: Loss {loss:.4f}")

        if step % config["training"]["save_steps"] == 0:
            model_engine.save_checkpoint("checkpoints", f"sft_{step}")

    print(f"SFT completed. Final checkpoint saved.")

    # Phase 2: Semi-supervised with Challenger-Answer
    print("\n=== Phase 2: Semi-Supervised Learning ===")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        print("Warning: OPENROUTER_API_KEY not set, skipping semi-supervised")
        return

    challenger = ChallengerAnswerGenerator(openrouter_key)
    semi_trainer = SemiSupervisedTrainer(model_engine, optimizer, config, challenger)

    topics = ["coding", "math", "science", "reasoning", "long-context"]

    for step in range(config["training"]["semi_supervised_steps"]):
        loss = semi_trainer.train_step(topics)

        if step % config["training"]["logging_steps"] == 0:
            print(f"Semi Step {step}: Loss {loss:.4f}")

        if step % config["training"]["save_steps"] == 0:
            model_engine.save_checkpoint("checkpoints", f"semi_{step}")

    print("Training completed!")

if __name__ == "__main__":
    main()
