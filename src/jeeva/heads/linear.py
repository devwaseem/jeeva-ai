from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from jeeva.heads.base import BaseDecisionHead


class LinearDecisionHead(BaseDecisionHead):
    """
    Dedicated linear decision probe for fixed-class categorical or binary decisions.

    Replaces static vocabulary projections (lm_head) with a trained linear mapping:
        z = (W * h + b) / temperature
    where each row w_k represents an optimal decision boundary separating classes
    directly within the transformer's hidden representation geometry.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int | None = None,
        labels: Sequence[str] | None = None,
        bias: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        if labels is not None:
            self.labels: list[str] | None = list(labels)
            num_classes = len(self.labels)
        else:
            self.labels = None

        if num_classes is None or num_classes < 1:
            raise ValueError(
                "Either `num_classes` (>=1) or a non-empty `labels` sequence must be provided."
            )

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = max(1e-4, float(temperature))
        self.proj = nn.Linear(hidden_dim, num_classes, bias=bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Computes calibrated decision logits for input representations.

        Args:
            h: Hidden representation tensor of shape [..., hidden_dim].

        Returns:
            Logits tensor of shape [..., num_classes].
        """
        raw_logits = self.proj(h.to(self.proj.weight.dtype))
        return raw_logits / self.temperature

    def predict(self, h: torch.Tensor) -> str | int | list[str | int]:
        """
        Computes predictions for input representations, mapping to labels if available.
        """
        with torch.no_grad():
            logits = self.forward(h)
            indices = torch.argmax(logits, dim=-1).tolist()

        if self.labels is not None:
            if isinstance(indices, list):
                return [self.labels[i] for i in indices]
            return self.labels[indices]
        return indices

    def get_config(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "labels": self.labels,
            "bias": self.proj.bias is not None,
            "temperature": self.temperature,
        }
