import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from jeeva._json import MAX_CONFIG_JSON_BYTES, read_json

MAX_HEAD_PARAMETERS = 64_000_000
MAX_HEAD_CLASSES = 4_096


def _positive_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Saved decision head {name!r} must be a positive integer.")
    return value


def _expected_tensor_shapes(
    head_type: str,
    config: dict[str, Any],
    *,
    expected_hidden_dim: int | None,
) -> dict[str, tuple[int, ...]]:
    hidden_dim = _positive_int(config, "hidden_dim")
    if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"Saved decision head hidden_dim={hidden_dim} does not match "
            f"backbone hidden_dim={expected_hidden_dim}."
        )

    shapes: dict[str, tuple[int, ...]]
    if head_type == "LinearDecisionHead":
        labels = config.get("labels")
        if labels is not None:
            if (
                not isinstance(labels, list)
                or not labels
                or not all(isinstance(label, str) for label in labels)
            ):
                raise ValueError("Saved linear-head labels must be a non-empty string list.")
            num_classes = len(labels)
        else:
            num_classes = _positive_int(config, "num_classes")
        if num_classes > MAX_HEAD_CLASSES:
            raise ValueError(
                f"Saved decision head has {num_classes} classes; maximum is {MAX_HEAD_CLASSES}."
            )
        bias = config.get("bias", True)
        if not isinstance(bias, bool):
            raise ValueError("Saved linear-head bias must be a boolean.")
        parameter_count = hidden_dim * num_classes + (num_classes if bias else 0)
        shapes = {"proj.weight": (num_classes, hidden_dim)}
        if bias:
            shapes["proj.bias"] = (num_classes,)
    elif head_type == "BilinearMetricHead":
        raw_rank = config.get("rank", 256)
        rank = hidden_dim if raw_rank is None else raw_rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("Saved metric-head rank must be a positive integer.")
        parameter_count = 2 * hidden_dim * rank + 1
        shapes = {
            "proj_query.weight": (rank, hidden_dim),
            "proj_key.weight": (rank, hidden_dim),
            "raw_temperature": (),
        }
    else:
        raise ValueError(f"Unsupported saved decision head type: {head_type!r}.")

    if parameter_count > MAX_HEAD_PARAMETERS:
        raise ValueError(
            f"Saved decision head requests {parameter_count:,} parameters; maximum is "
            f"{MAX_HEAD_PARAMETERS:,}."
        )
    return shapes


def _validate_tensor_shapes(
    weights_path: Path,
    expected_shapes: dict[str, tuple[int, ...]],
) -> None:
    with safe_open(weights_path, framework="pt", device="cpu") as tensors:
        actual_shapes = {
            name: tuple(tensors.get_slice(name).get_shape()) for name in tensors.keys()
        }
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Saved decision-head tensor shapes do not match config: "
            f"expected {expected_shapes}, received {actual_shapes}."
        )


class BaseDecisionHead(nn.Module, ABC):
    """
    Abstract base class for learned decision projection heads in Jeeva.
    Replaces static vocabulary projections with trained task-specific decision boundaries.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Computes unnormalized decision logits or metric matching scores."""
        pass

    def save_pretrained(self, save_dir: str | Path) -> None:
        """Saves the head weights and architecture configuration to disk."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = self.get_config()
        config["head_type"] = self.__class__.__name__
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        save_file(self.state_dict(), path / "model.safetensors")

    @classmethod
    def from_pretrained(
        cls,
        load_dir: str | Path,
        map_location: str | None = None,
        *,
        expected_hidden_dim: int | None = None,
    ) -> "BaseDecisionHead":
        """Load a saved head after validating its allocation and tensor shapes."""
        path = Path(load_dir)
        loaded_config = read_json(
            path / "config.json",
            max_bytes=MAX_CONFIG_JSON_BYTES,
        )
        if not isinstance(loaded_config, dict):
            raise ValueError("Saved decision-head config must be a JSON object.")
        config = dict(loaded_config)

        head_type = config.pop("head_type", cls.__name__)
        if not isinstance(head_type, str):
            raise ValueError("Saved decision-head type must be a string.")
        expected_shapes = _expected_tensor_shapes(
            head_type,
            config,
            expected_hidden_dim=expected_hidden_dim,
        )
        weights_path = path / "model.safetensors"
        _validate_tensor_shapes(weights_path, expected_shapes)

        from jeeva.heads.linear import LinearDecisionHead
        from jeeva.heads.metric import BilinearMetricHead

        head_classes = {
            "LinearDecisionHead": LinearDecisionHead,
            "BilinearMetricHead": BilinearMetricHead,
        }
        target_cls = head_classes[head_type]
        instance = target_cls(**config)
        state_dict = load_file(weights_path, device=map_location or "cpu")
        instance.load_state_dict(state_dict, strict=True)
        return instance

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Returns initialization configuration dictionary for serialization."""
        pass
