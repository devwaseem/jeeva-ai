import math
from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_finite_value(value: float, *, name: str, context: str) -> None:
    """Stop a training run before a non-finite value can reach a checkpoint."""
    if math.isfinite(value):
        return
    raise FloatingPointError(f"Training stopped: non-finite {name} {value} at {context}.")


def require_finite_loss(loss: torch.Tensor, *, context: str) -> None:
    """Stop a training run before backpropagation when its scalar loss diverges."""
    value = float(loss.detach().to(device="cpu", dtype=torch.float32).item())
    require_finite_value(value, name="loss", context=context)


def backward_finite_loss(loss: torch.Tensor, *, context: str) -> None:
    """Backpropagate a finite loss, or stop before gradients can be contaminated."""
    require_finite_loss(loss, context=context)
    loss.backward()


def clip_finite_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float,
    context: str,
) -> torch.Tensor:
    """Clip finite gradients and stop before an invalid optimizer update."""
    effective_max_norm = max_norm if max_norm > 0 else float("inf")
    try:
        return torch.nn.utils.clip_grad_norm_(
            list(parameters),
            effective_max_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        raise FloatingPointError(
            f"Training stopped: non-finite gradient norm at {context}."
        ) from exc


class BrierScoreLoss(nn.Module):
    """
    Multi-Class Brier Score loss function for calibrated LoRA fine-tuning.

    Computes:
        Brier Loss = 1/K * sum_{k=1}^K (P_k - Y_k)^2

    Supports:
    - Raw logits (from_logits=True) or probabilities (from_logits=False)
    - One-hot integer targets or empirical soft distributions (e.g. [0.5, 0.4, 0.1])
    - Reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        temperature: float = 1.0,
        from_logits: bool = True,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.reduction = reduction
        self.temperature = temperature
        self.from_logits = from_logits

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape [..., K] (sliced logits or probabilities)
            target: Tensor of shape [..., K] (distribution) or [...] (class indices)
        """
        if self.from_logits:
            probs = F.softmax(input / self.temperature, dim=-1)
        else:
            probs = input

        # Convert class indices to one-hot if necessary
        if target.shape != probs.shape:
            if target.dtype in (torch.int64, torch.int32, torch.long):
                target = F.one_hot(target, num_classes=probs.shape[-1]).to(
                    dtype=probs.dtype, device=probs.device
                )
            else:
                raise ValueError(
                    f"Target shape {target.shape} does not match input shape {probs.shape} "
                    "and cannot be converted as integer class indices."
                )
        else:
            target = target.to(dtype=probs.dtype, device=probs.device)

        # 1/K * sum_{k=1}^K (P_k - Y_k)^2
        per_sample_loss = torch.mean((probs - target).square(), dim=-1)

        if self.reduction == "mean":
            return torch.mean(per_sample_loss)
        elif self.reduction == "sum":
            return torch.sum(per_sample_loss)
        return per_sample_loss


class CompositeDecisionLoss(nn.Module):
    """Soft cross-entropy plus calibration and optional ordinal distribution loss."""

    def __init__(
        self,
        *,
        brier_weight: float = 0.5,
        ordinal_weight: float = 0.25,
        temperature: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        if brier_weight < 0.0 or ordinal_weight < 0.0:
            raise ValueError("Loss weights cannot be negative.")
        if temperature <= 0.0:
            raise ValueError("Temperature must be positive.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.brier_weight = brier_weight
        self.ordinal_weight = ordinal_weight
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        level_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        if target.shape != probabilities.shape:
            if target.dtype in (torch.int64, torch.int32, torch.long):
                target = F.one_hot(target, num_classes=probabilities.shape[-1])
            else:
                raise ValueError(
                    f"Target shape {target.shape} does not match logits shape {logits.shape}."
                )
        target = target.to(dtype=probabilities.dtype, device=probabilities.device)

        log_probabilities = F.log_softmax(logits / self.temperature, dim=-1)
        cross_entropy = -(target * log_probabilities).sum(dim=-1)
        brier = torch.mean((probabilities - target).square(), dim=-1)
        per_sample = cross_entropy + self.brier_weight * brier

        if level_values is not None and self.ordinal_weight > 0.0:
            levels = torch.as_tensor(
                level_values,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )
            if levels.ndim != 1 or levels.numel() != probabilities.shape[-1]:
                raise ValueError("level_values must match the logits' candidate dimension.")
            intervals = torch.diff(levels)
            if not torch.all(intervals > 0):
                raise ValueError("level_values must be strictly increasing.")
            interval_weights = intervals / intervals.sum()
            predicted_cdf = torch.cumsum(probabilities, dim=-1)[..., :-1]
            target_cdf = torch.cumsum(target, dim=-1)[..., :-1]
            ordinal = torch.sum(
                interval_weights * (predicted_cdf - target_cdf).square(),
                dim=-1,
            )
            per_sample = per_sample + self.ordinal_weight * ordinal

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample
