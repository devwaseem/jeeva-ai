from __future__ import annotations

from pathlib import Path

from jeeva._json import MAX_CONFIG_JSON_BYTES, MAX_JSON_BYTES, read_json

ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
MODEL_SAFE_WEIGHTS_NAMES = ("model.safetensors", "model.safetensors.index.json")


def require_local_adapter_safetensors(source: str | Path) -> None:
    """Reject local PEFT adapters that could fall back to pickle-backed weights."""
    path = Path(source)
    if not (path / ADAPTER_SAFE_WEIGHTS_NAME).is_file():
        raise ValueError(
            f"Adapter at {path} must contain {ADAPTER_SAFE_WEIGHTS_NAME}; "
            "pickle-backed adapter weights are not supported."
        )
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Adapter at {path} must contain adapter_config.json.")
    config = read_json(config_path, max_bytes=MAX_CONFIG_JSON_BYTES)
    if not isinstance(config, dict):
        raise ValueError(f"Adapter config at {config_path} must be a JSON object.")


def require_local_model_safetensors(source: str | Path) -> None:
    """Require a bounded config and standard Safetensors model artifact."""
    path = Path(source)
    if not any((path / name).is_file() for name in MODEL_SAFE_WEIGHTS_NAMES):
        expected = " or ".join(MODEL_SAFE_WEIGHTS_NAMES)
        raise ValueError(
            f"Model artifact at {path} must contain {expected}; "
            "pickle-backed model weights are not supported."
        )
    config_path = path / "config.json"
    if config_path.is_file():
        config = read_json(config_path, max_bytes=MAX_CONFIG_JSON_BYTES)
        if not isinstance(config, dict):
            raise ValueError(f"Model config at {config_path} must be a JSON object.")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = read_json(index_path, max_bytes=MAX_JSON_BYTES)
        if not isinstance(index, dict):
            raise ValueError(f"Model index at {index_path} must be a JSON object.")
