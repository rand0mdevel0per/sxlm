"""Ring Buffer: Monitor attention entropy and trigger replan"""

import torch
from collections import deque

class RingBuffer:
    def __init__(self, buffer_size: int = 128):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def push(self, snapshot: dict):
        """Store planning snapshot"""
        self.buffer.append(snapshot)

    def detect_drift(self, current_state: torch.Tensor,
                     attention_weights: torch.Tensor,
                     effort_threshold: float) -> bool:
        """
        Detect logical drift using attention entropy + cosine similarity

        Args:
            current_state: Current hidden state [batch, seq_len, dim]
            attention_weights: Current attention weights [batch, heads, seq, seq]
            effort_threshold: Dynamic threshold from Planner-Port

        Returns:
            True if drift detected, False otherwise
        """
        if len(self.buffer) == 0:
            return False

        # Calculate attention entropy
        entropy = self._attention_entropy(attention_weights)

        # Calculate cosine similarity with recent snapshots
        similarity = self._cosine_similarity(current_state)

        # Dynamic threshold adjustment
        theta = effort_threshold * 0.5  # Base threshold scaled by effort

        # Drift detection: high entropy OR low similarity
        drift = (entropy > theta) or (similarity < 0.7)

        return drift

    def _attention_entropy(self, attn_weights: torch.Tensor) -> float:
        """Calculate attention entropy (higher = more uncertain)"""
        # attn_weights: [batch, heads, seq, seq]
        probs = attn_weights.mean(dim=(0, 1))  # Average over batch and heads
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        return entropy.item()

    def _cosine_similarity(self, current_state: torch.Tensor) -> float:
        """Calculate cosine similarity with recent snapshots"""
        if len(self.buffer) == 0:
            return 1.0

        recent_snapshot = self.buffer[-1]
        past_state = recent_snapshot.get('hidden_state')

        if past_state is None:
            return 1.0

        # Cosine similarity
        current_flat = current_state.mean(dim=1)  # [batch, dim]
        past_flat = past_state.mean(dim=1)

        similarity = torch.cosine_similarity(current_flat, past_flat, dim=-1)
        return similarity.mean().item()
