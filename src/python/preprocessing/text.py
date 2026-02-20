"""Text preprocessing for VQ-GAN encoding"""

import torch
from typing import List

class TextPreprocessor:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs"""
        return [hash(word) % self.vocab_size for word in text.split()]

    def preprocess(self, text: str) -> torch.Tensor:
        """Preprocess text for VQ-GAN encoding"""
        tokens = self.tokenize(text)
        return torch.tensor(tokens, dtype=torch.long)

    def batch_preprocess(self, texts: List[str]) -> torch.Tensor:
        """Batch preprocess multiple texts"""
        return torch.stack([self.preprocess(t) for t in texts])
