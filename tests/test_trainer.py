import importlib.util

import pytest
import torch

from jeeva.core.engine import DecisionEngine
from jeeva.data.dataset import JeevaDataset
from jeeva.heads.metric import DECODER_HEAD_DIR
from jeeva.training.trainer import Trainer, TrainingConfig, _epoch_sample_order
from tests.schema_helpers import choice, noul, schema

_ACCELERATOR_DEVICE = (
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)
_ACCELERATOR_DTYPE = "float16" if _ACCELERATOR_DEVICE != "cpu" else "float32"


@pytest.fixture
def mini_dataset():
    decision_schema = schema(
        cat=choice(["A", "B"], criterion="Pick category."),
        urg=noul("The sample is urgent."),
    )
    records = [
        {"text": "Sample 1 billing inquiry", "cat": "A", "urg": True},
        {"text": "Sample 2 technical issue", "cat": "B", "urg": False},
        {"text": "Sample 3 missing label", "urg": True},  # Missing cat
    ]
    return JeevaDataset(data=records, schema=decision_schema), decision_schema


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.epochs == 5
    assert cfg.backprop_through_prefill is True
    assert cfg.contiguous_cache is True
    assert cfg.lr == 2e-4
    assert cfg.lora_r == 8
    assert cfg.progress_updates_per_epoch == 10
    assert cfg.decoder_head == "letter"
    assert TrainingConfig(decoder_head="contextual").decoder_head == "contextual"
    assert TrainingConfig(adaptation="qlora").adaptation == "qlora"
    assert cfg.head_lr == pytest.approx(1e-3)
    assert cfg.candidate_position_balancing is True
    assert cfg.brier_weight == 0.5
    assert cfg.ordinal_weight == 0.25

    with pytest.raises(ValueError, match="progress_updates_per_epoch"):
        TrainingConfig(progress_updates_per_epoch=-1)
    with pytest.raises(ValueError, match="decoder_head"):
        TrainingConfig(decoder_head="dense")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decoder_head"):
        TrainingConfig(decoder_head="model")  # type: ignore[arg-type]


def test_epoch_sample_order_is_shuffled_and_reproducible():
    sequential = list(range(20))
    first_epoch = _epoch_sample_order(20, seed=42, epoch=1)
    second_epoch = _epoch_sample_order(20, seed=42, epoch=2)

    assert sorted(first_epoch) == sequential
    assert first_epoch != sequential
    assert second_epoch != first_epoch
    assert _epoch_sample_order(20, seed=42, epoch=1) == first_epoch


@pytest.mark.training_integration
def test_trainer_step_both_prefill_modes(mini_dataset, tmp_path, capsys):
    dataset, schema = mini_dataset

    # Test backprop_through_prefill=True
    cfg_true = TrainingConfig(
        epochs=1,
        lr=1e-3,
        backprop_through_prefill=True,
        output_dir=str(tmp_path / "ckpt_true"),
        progress_updates_per_epoch=2,
    )
    trainer_true = Trainer(
        model_or_id="Qwen/Qwen2.5-0.5B-Instruct",
        schema=schema,
        config=cfg_true,
    )
    res_true = trainer_true.train(train_dataset=dataset)
    output = capsys.readouterr().out
    assert "Training started | samples=3 | validation=0 | epochs=1" in output
    assert "Epoch 01/01 started | samples=3" in output
    assert "Progress 2/3 (66.7%)" in output
    assert "Progress 3/3 (100.0%)" in output
    assert "loss=" in output
    assert "elapsed=" in output
    assert "eta=" in output
    assert len(res_true["history"]) == 1
    assert res_true["scorer"] == "letter"
    assert (tmp_path / "ckpt_true" / "adapter_config.json").exists()

    # Test backprop_through_prefill=False (detached KV-cache memory-saving mode)
    cfg_false = TrainingConfig(
        epochs=1,
        lr=1e-3,
        backprop_through_prefill=False,
        output_dir=str(tmp_path / "ckpt_false"),
    )
    trainer_false = Trainer(
        model_or_id="Qwen/Qwen2.5-0.5B-Instruct",
        schema=schema,
        config=cfg_false,
    )
    res_false = trainer_false.train(train_dataset=dataset)
    assert len(res_false["history"]) == 1
    assert (tmp_path / "ckpt_false" / "adapter_config.json").exists()
    assert res_false["scorer"] == "letter"


@pytest.mark.training_integration
def test_trainer_step_bilinear_head(mini_dataset, tmp_path):
    dataset, decision_schema = mini_dataset
    output_dir = tmp_path / "ckpt_bilinear"
    trainer = Trainer(
        model_or_id="Qwen/Qwen2.5-0.5B-Instruct",
        schema=decision_schema,
        config=TrainingConfig(
            epochs=1,
            lr=1e-3,
            decoder_head="bilinear",
            head_lr=1e-3,
            output_dir=str(output_dir),
            progress_updates_per_epoch=0,
            device=_ACCELERATOR_DEVICE,
            dtype=_ACCELERATOR_DTYPE,
        ),
    )

    report = trainer.train(train_dataset=dataset)

    assert report["scorer"] == "bilinear"
    assert torch.isfinite(torch.tensor(report["history"][0]["train_loss"]))
    assert (output_dir / DECODER_HEAD_DIR / "config.json").is_file()
    assert (output_dir / DECODER_HEAD_DIR / "model.safetensors").is_file()


@pytest.mark.training_integration
def test_trainer_step_contextual_head_scores_contextual_candidates(mini_dataset, tmp_path):
    dataset, decision_schema = mini_dataset
    output_dir = tmp_path / "ckpt_contextual"
    trainer = Trainer(
        model_or_id="Qwen/Qwen2.5-0.5B-Instruct",
        schema=decision_schema,
        config=TrainingConfig(
            epochs=1,
            lr=1e-3,
            decoder_head="contextual",
            head_lr=1e-3,
            output_dir=str(output_dir),
            progress_updates_per_epoch=0,
            device=_ACCELERATOR_DEVICE,
            dtype=_ACCELERATOR_DTYPE,
        ),
    )

    report = trainer.train(train_dataset=dataset)
    assert trainer.head is not None
    assert trainer.head.input_strategy == "contextual"
    trainer.model.eval()
    trainer.head.eval()
    engine = DecisionEngine.from_components(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        processor=trainer.processor,
        model_id=trainer.model_id,
        device=trainer.device,
        dtype=trainer.torch_dtype,
        mode="dense",
        head=trainer.head,
        candidate_position_balancing=False,
    )
    result = engine.evaluate("Sample 1 billing inquiry", decision_schema)

    assert report["scorer"] == "contextual"
    assert torch.isfinite(torch.tensor(report["history"][0]["train_loss"]))
    assert (output_dir / DECODER_HEAD_DIR / "config.json").is_file()
    assert set(result.decisions) == {"cat", "urg"}
    assert all(decision.distribution for decision in result.decisions.values())


_QLORA_DEVICE = (
    "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)
)


@pytest.mark.training_integration
@pytest.mark.skipif(
    _QLORA_DEVICE is None or importlib.util.find_spec("bitsandbytes") is None,
    reason="QLoRA integration requires bitsandbytes and an MPS or CUDA device.",
)
def test_trainer_step_qlora_on_accelerator(mini_dataset, tmp_path):
    dataset, decision_schema = mini_dataset
    output_dir = tmp_path / "ckpt_qlora"
    trainer = Trainer(
        model_or_id="Qwen/Qwen2.5-0.5B-Instruct",
        schema=decision_schema,
        config=TrainingConfig(
            epochs=1,
            lr=1e-3,
            adaptation="qlora",
            output_dir=str(output_dir),
            progress_updates_per_epoch=0,
            device=_QLORA_DEVICE,
            dtype="float16",
        ),
    )

    report = trainer.train(train_dataset=dataset)

    assert report["adaptation"] == "qlora"
    assert report["quantization"] == {
        "backend": "bitsandbytes",
        "backend_version": report["quantization"]["backend_version"],
        "bits": 4,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "float16",
    }
    assert (output_dir / "adapter_config.json").is_file()
