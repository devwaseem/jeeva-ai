from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from jeeva.heads.base import BaseDecisionHead

DECODER_HEAD_DIR = "decoder_head"

MetricInputStrategy = Literal["static", "contextual"]


class BilinearMetricHead(BaseDecisionHead):
    """Normalized metric head for dynamic, open-set candidate options.

    ``static`` inputs compare a contextual query against context-free option
    embeddings. ``contextual`` inputs compare a query branch against isolated
    candidate branches that share the same document prefill.

    Both strategies project query and candidates into an aligned metric space:
        q = Normalize(W_q * h_query)
        k_i = Normalize(W_k * e(o_i))
        score_i = (q · k_i) / temperature
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int | None = 256,
        temperature: float = 1.0,
        learnable_temperature: bool = True,
        input_strategy: MetricInputStrategy = "static",
    ):
        super().__init__()
        if input_strategy not in ("static", "contextual"):
            raise ValueError("input_strategy must be 'static' or 'contextual'.")
        self.hidden_dim = hidden_dim
        self.rank = rank or hidden_dim
        self.learnable_temperature = learnable_temperature
        self.input_strategy = input_strategy

        self.proj_query = nn.Linear(hidden_dim, self.rank, bias=False)
        self.proj_key = nn.Linear(hidden_dim, self.rank, bias=False)

        if learnable_temperature:
            self.raw_temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer("raw_temperature", torch.tensor(float(temperature)))

    @property
    def temperature(self) -> float:
        return float(torch.clamp(self.raw_temperature, min=1e-4).item())

    def forward(
        self,
        h_query: torch.Tensor,
        h_options: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes calibrated metric similarity scores between query lane(s) and candidate options.

        Args:
            h_query: Query representation tensor of shape [..., hidden_dim] or [batch, hidden_dim].
            h_options: Candidate option embeddings tensor of shape [num_options, hidden_dim].

        Returns:
            Scores tensor of shape [..., num_options].
        """
        target_dtype = self.proj_query.weight.dtype
        q = F.normalize(self.proj_query(h_query.to(target_dtype)), p=2, dim=-1)
        k = F.normalize(self.proj_key(h_options.to(target_dtype)), p=2, dim=-1)

        temp = torch.clamp(self.raw_temperature, min=1e-4)
        scores = torch.matmul(q, k.T) / temp
        return scores

    def get_config(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "temperature": self.temperature,
            "learnable_temperature": self.learnable_temperature,
            "input_strategy": self.input_strategy,
        }
