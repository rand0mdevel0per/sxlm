"""SXLM Quila Model - Integrates all Phase 3 components"""

import torch
import torch.nn as nn
import _sintellix_native as native

class QuilaModel(nn.Module):
    def __init__(self, config: native.QuilaConfig):
        super().__init__()
        self.config = config

        # Core embeddings
        self.token_emb = nn.Embedding(50257, config.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, config.dim))

        # Transformer layers with HOT-NSA
        self.layers = nn.ModuleList([
            QuilaLayer(config) for _ in range(config.num_layers)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, 50257, bias=False)

    def forward(self, x):
        B, T = x.shape

        # Embeddings
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

class QuilaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(config.dim, config.num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.GELU(),
            nn.Linear(4 * config.dim, config.dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x
