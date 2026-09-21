import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from jeeva._weights import (
    require_local_adapter_safetensors,
    require_local_model_safetensors,
)
from jeeva.core.engine import (
    DecisionEngine,
    _load_causal_or_vision_model,
    _subtract_score_offset,
    config_declares_multimodal,
)
from tests.schema_helpers import choice, noul


def test_from_components_does_not_load_another_checkpoint():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"

    with (
        patch("jeeva.core.engine.AutoTokenizer.from_pretrained") as load_tokenizer,
        patch.object(DecisionEngine, "_precache_letter_weights"),
    ):
        engine = DecisionEngine.from_components(
            model=model,
            tokenizer=tokenizer,
            model_id="already-loaded",
            device="cpu",
            dtype=torch.float32,
            contiguous_cache=False,
            candidate_position_balancing=True,
        )

    load_tokenizer.assert_not_called()
    assert engine.model is model
    assert engine.tokenizer is tokenizer
    assert engine.contiguous_cache is False
    assert engine.torch_dtype == torch.float32
    assert engine.dtype == torch.float32
    assert engine.candidate_position_balancing is True
    assert not hasattr(engine, "_branch_inputs_cache")
    assert not hasattr(engine, "_option_embedding_cache")


def test_model_loading_requires_safetensors():
    model = MagicMock()
    model.to.return_value = model
    with patch(
        "jeeva.core.engine.AutoModelForCausalLM.from_pretrained",
        return_value=model,
    ) as load_model:
        loaded = _load_causal_or_vision_model(
            "model-id",
            device="cpu",
            dtype=torch.float32,
        )

    assert loaded is model
    assert load_model.call_args.kwargs["use_safetensors"] is True


def test_local_adapter_requires_safetensors(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        require_local_adapter_safetensors(tmp_path)


def test_local_model_requires_safetensors(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="model.safetensors"):
        require_local_model_safetensors(tmp_path)


def test_multimodal_config_detection_requires_explicit_image_metadata():
    assert config_declares_multimodal(SimpleNamespace()) is False
    assert config_declares_multimodal(SimpleNamespace(vision_config=SimpleNamespace())) is True


def test_text_model_skips_processor_loading():
    model = MagicMock()
    model.config = SimpleNamespace(
        _name_or_path="text-model",
        _commit_hash="model-commit",
    )
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.name_or_path = "text-model"
    tokenizer.init_kwargs = {"_commit_hash": "tokenizer-commit"}
    with (
        patch("jeeva.core.engine.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "jeeva.core.engine._load_causal_or_vision_model",
            return_value=model,
        ),
        patch("transformers.AutoProcessor.from_pretrained") as load_processor,
    ):
        engine = DecisionEngine("text-model", device="cpu", mode="dense")
    load_processor.assert_not_called()
    assert engine.processor is None
    assert engine.is_multimodal is False


def test_multimodal_processor_load_failures_are_not_hidden():
    model = MagicMock()
    model.config = SimpleNamespace(vision_config=SimpleNamespace())
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    with (
        patch("jeeva.core.engine.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "jeeva.core.engine._load_causal_or_vision_model",
            return_value=model,
        ),
        patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=OSError("unauthorized processor"),
        ),
        pytest.raises(OSError, match="unauthorized processor"),
    ):
        DecisionEngine("vision-model", device="cpu", mode="dense")


def test_global_readout_mode_applies_to_every_lane():
    engine = DecisionEngine.__new__(DecisionEngine)
    engine.mode = "letter"
    primitives = [
        noul("The text is negative."),
        choice(["A", "B"]),
    ]
    assert engine._resolve_readout_modes(primitives, mode=None) == ["letter", "letter"]
    assert engine._resolve_readout_modes(primitives, mode="dense") == ["dense", "dense"]
    with pytest.raises(ValueError, match="Unsupported decision mode 'in_context'"):
        engine._resolve_readout_modes(primitives, mode="in_context")


def test_score_offset_is_subtracted_before_probability_parsing():
    raw = torch.tensor([4.0, 2.0])

    calibrated = _subtract_score_offset(raw, (3.0, 0.0), field_name="answer")

    assert torch.equal(calibrated, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="has 3 scores; expected 2"):
        _subtract_score_offset(raw, (1.0, 2.0, 3.0), field_name="answer")


def test_letter_weight_uses_bare_completion_token_after_prompt_space():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
        " A": [2],
        "A": [1],
    }[text]

    model = MagicMock()
    model.lm_head.weight = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )

    engine = DecisionEngine.__new__(DecisionEngine)
    engine.tokenizer = tokenizer
    engine.model = model
    engine.device = "cpu"
    engine._letter_weights = {}

    weight = engine._get_letter_weight("A")

    assert torch.equal(weight, torch.tensor([1.0, 1.0]))
    tokenizer.encode.assert_called_once_with("A", add_special_tokens=False)


def test_letter_weight_rejects_non_single_bare_candidate():
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
        " A": [2],
        "A": [1, 3],
    }[text]

    engine = DecisionEngine.__new__(DecisionEngine)
    engine.model = MagicMock()
    engine.model.lm_head.weight = torch.zeros(4, 2)
    engine.device = "cpu"
    engine.tokenizer = tokenizer
    engine.model_id = "test-model"
    engine._letter_weights = {}

    with pytest.raises(ValueError, match="Letter candidate"):
        engine._get_letter_weight("A")

    tokenizer.encode.assert_called_once_with("A", add_special_tokens=False)


def test_public_import_exposes_decision_schema_api_without_training_internals():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, jeeva, jeeva.core, jeeva.training; "
                "assert 'jeeva.training.trainer' not in sys.modules; "
                "assert jeeva.DecisionModel is not None; "
                "assert jeeva.DecisionSchema is not None; "
                "assert jeeva.Noul is not None; "
                "assert not hasattr(jeeva, 'DecisionEngine'); "
                "assert not hasattr(jeeva.core, 'DecisionEngine'); "
                "assert not hasattr(jeeva.training, 'Trainer')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
