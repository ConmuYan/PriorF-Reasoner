"""Binary classification head for fraud detection.

A simple single-layer classification head that maps the last-token
hidden state from the LLM to a single logit for binary classification
(fraud vs. benign).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryClsHead(nn.Module):
    """Binary classification head for fraud prediction.

    Takes a hidden-state vector (from the last non-padding token) and
    produces a single logit. The logit can be passed through sigmoid
    to obtain a probability, or used directly with BCEWithLogitsLoss.

    Args:
        hidden_dim: Dimension of the input hidden state.
        dropout: Dropout probability before the linear layer.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden: Hidden state tensor of shape ``(batch, hidden_dim)``.

        Returns:
            Logits tensor of shape ``(batch,)`` (squeezed from (batch, 1)).
        """
        x = self.dropout(hidden)
        x = x.to(dtype=self.linear.weight.dtype)
        logits = self.linear(x).squeeze(-1)  # (batch,)
        return logits

    def predict_proba(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute fraud probability via sigmoid.

        Args:
            hidden: Hidden state tensor ``(batch, hidden_dim)``.

        Returns:
            Probabilities tensor of shape ``(batch,)`` in [0, 1].
        """
        logits = self.forward(hidden)
        return torch.sigmoid(logits)
