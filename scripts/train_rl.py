"""RL training script with eligibility traces"""

import sys
sys.path.insert(0, 'E:/sxlm/python')

import torch
from sxlm.training import RLTrainer
import _sintellix_native as sxlm

def main():
    # Load config
    config = sxlm.QuilaConfig.load("config.toml")

    # Create model
    model = torch.nn.Linear(config.dim, 256).cuda()

    # Train
    trainer = RLTrainer(model, learning_rate=config.learning_rate,
                       trace_decay=config.el_trace_decay)

    for episode in range(100):
        state = torch.randn(1, config.dim).cuda()
        action = torch.randint(0, 256, (1,)).cuda()
        reward = 1.0
        next_state = torch.randn(1, config.dim).cuda()

        loss = trainer.train_step(state, action, reward, next_state)

        if episode % 10 == 0:
            print(f"Episode {episode}: Loss = {loss:.4f}")

if __name__ == "__main__":
    main()
