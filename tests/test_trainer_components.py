import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BatchEncoding

import jeeva.training.trainer as trainer_module
from jeeva.training.trainer import (
    Trainer,
    TrainingConfig,
    _load_qlora_causal_model,
    _require_qlora_dependencies,
)
from tests.schema_helpers import choice, schema


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        count = len(texts)
        return BatchEncoding(
            {
                "input_ids": torch.arange(count, dtype=torch.long).unsqueeze(1),
                "attention_mask": torch.ones((count, 1), dtype=torch.long),
            }
        )

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(text)]


class DropoutEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, 4)
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, input_ids, **kwargs):
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(hidden_states=[hidden])


def test_option_embeddings_are_deterministic_and_restore_training_mode():
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.tokenizer = FakeTokenizer()
    trainer.model = DropoutEncoder()
    trainer.model.train()
    trainer._option_embedding_cache = {}
    primitive = choice(["A", "B"])

    first = trainer._get_or_compute_option_embeddings(primitive)
    trainer._option_embedding_cache.clear()
    second = trainer._get_or_compute_option_embeddings(primitive)

    assert trainer.model.training is True
    assert torch.equal(first, second)


def test_letter_scorer_builds_balanced_letter_branches():
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(device="cpu")
    trainer.primitives = [choice(["Yes", "No"])]
    trainer.tokenizer = FakeTokenizer()
    trainer._branch_suffix = "\n\nAssistant:\n"

    prompts, branch_indices, orders, token_ids = trainer._build_training_branches()

    assert len(prompts) == 2
    assert all(prompt.endswith("Assistant:\nAnswer: ") for prompt in prompts)
    assert branch_indices == [[0, 1]]
    assert orders == [[(0, 1), (1, 0)]]
    assert token_ids == [(ord("A"), ord("B"))]


def test_bilinear_decoder_head_builds_one_semantic_branch():
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(device="cpu", decoder_head="bilinear")
    trainer.primitives = [choice(["Yes", "No"])]
    trainer.tokenizer = FakeTokenizer()
    trainer._branch_suffix = "\n\nAssistant:\n"

    prompts, branch_indices, orders, token_ids = trainer._build_training_branches()

    assert len(prompts) == 1
    assert prompts[0].startswith("\n\nAssistant:\n")
    assert not prompts[0].endswith("Assistant:\nAnswer: ")
    assert branch_indices == [[0]]
    assert orders == [[]]
    assert token_ids == [None]


def test_contextual_decoder_head_builds_query_and_isolated_candidate_branches():
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(device="cpu", decoder_head="contextual")
    trainer.primitives = [choice(["Yes-only", "No-only"])]
    trainer.tokenizer = FakeTokenizer()
    trainer._branch_suffix = "\n\nAssistant:\n"

    prompts, branch_indices, orders, token_ids = trainer._build_training_branches()

    assert len(prompts) == 3
    assert all(prompt.startswith("\n\nAssistant:\n") for prompt in prompts)
    assert "Yes-only" not in prompts[0]
    assert "No-only" not in prompts[0]
    assert "Yes-only" in prompts[1]
    assert "No-only" not in prompts[1]
    assert "No-only" in prompts[2]
    assert "Yes-only" not in prompts[2]
    assert branch_indices == [[0, 1, 2]]
    assert orders == [[]]
    assert token_ids == [None]


class TrainableAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class ConfiguredAdapter(TrainableAdapter):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def print_trainable_parameters(self) -> None:
        pass


class QLoRAAdapter(TrainableAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.to_called = False

    def to(self, *args, **kwargs):
        self.to_called = True
        return super().to(*args, **kwargs)

    def print_trainable_parameters(self) -> None:
        pass


@pytest.mark.parametrize("decoder_head", ["bilinear", "contextual"])
def test_metric_head_stays_float32_with_float16_backbone(
    decoder_head,
    monkeypatch,
):
    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token=None,
        eos_token="<eos>",
    )
    adapter = ConfiguredAdapter(hidden_size=8)

    monkeypatch.setattr(
        trainer_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        trainer_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(
        trainer_module,
        "_load_causal_or_vision_model",
        lambda **_kwargs: nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        trainer_module,
        "get_peft_model",
        lambda _base_model, _config: adapter,
    )

    trainer = Trainer(
        model_or_id="base-model",
        schema=schema(answer=choice(["Yes", "No"])),
        config=TrainingConfig(
            device="cpu",
            dtype="float16",
            decoder_head=decoder_head,
        ),
    )

    assert trainer.torch_dtype == torch.float16
    assert trainer.head is not None
    assert {parameter.dtype for parameter in trainer.head.parameters()} == {torch.float32}


def test_qlora_loader_builds_nf4_and_prepares_without_checkpointing(monkeypatch):
    quantization_kwargs = {}
    load_kwargs = {}
    prepare_kwargs = {}
    model = nn.Linear(1, 1)

    def build_quantization_config(**kwargs):
        quantization_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    def load_model(*, pretrained_model_name_or_path, **kwargs):
        load_kwargs.update(model=pretrained_model_name_or_path, **kwargs)
        return model

    def prepare_model(base_model, **kwargs):
        prepare_kwargs.update(model=base_model, **kwargs)
        return base_model

    monkeypatch.setattr(trainer_module.metadata, "version", lambda _name: "0.50.2")
    monkeypatch.setattr(trainer_module, "BitsAndBytesConfig", build_quantization_config)
    monkeypatch.setattr(
        trainer_module.AutoModelForCausalLM,
        "from_pretrained",
        load_model,
    )
    monkeypatch.setattr(trainer_module, "prepare_model_for_kbit_training", prepare_model)

    loaded, quantization = _load_qlora_causal_model(
        model_name_or_path="base-model",
        device="mps",
        dtype=torch.float16,
    )

    assert loaded is model
    assert quantization_kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.float16,
    }
    assert load_kwargs["model"] == "base-model"
    assert load_kwargs["device_map"] == {"": "mps"}
    assert load_kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"
    assert load_kwargs["use_safetensors"] is True
    assert prepare_kwargs == {
        "model": model,
        "use_gradient_checkpointing": False,
    }
    assert quantization == {
        "backend": "bitsandbytes",
        "backend_version": "0.50.2",
        "bits": 4,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "float16",
    }


def test_qlora_dependency_error_names_install_extra(monkeypatch):
    def missing(_name):
        raise trainer_module.metadata.PackageNotFoundError

    monkeypatch.setattr(trainer_module.metadata, "version", missing)

    with pytest.raises(ImportError, match=r"jeeva-ai\[qlora\]"):
        _require_qlora_dependencies()


def test_qlora_trainer_rejects_cpu_and_preconstructed_models():
    decision_schema = schema(answer=choice(["Yes", "No"]))

    with pytest.raises(ValueError, match="MPS or CUDA"):
        Trainer(
            model_or_id="base-model",
            schema=decision_schema,
            config=TrainingConfig(device="cpu", adaptation="qlora"),
        )

    with pytest.raises(ValueError, match="model identifier"):
        Trainer(
            model_or_id=nn.Linear(1, 1),
            schema=decision_schema,
            config=TrainingConfig(device="mps", adaptation="qlora"),
        )


def test_qlora_live_export_requires_unquantized_artifact_merge(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(device="mps", adaptation="qlora")
    destination = tmp_path / "merged"

    with pytest.raises(ValueError, match="live 4-bit QLoRA model"):
        trainer.export_merged(destination)

    assert not destination.exists()


def test_qlora_trainer_attaches_all_linear_adapter_without_model_move(monkeypatch):
    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token=None,
        eos_token="<eos>",
    )
    adapter = QLoRAAdapter()
    attached = {}
    quantization = {
        "backend": "bitsandbytes",
        "backend_version": "0.50.2",
        "bits": 4,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "float16",
    }

    monkeypatch.setattr(
        trainer_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        trainer_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(
        trainer_module,
        "_load_qlora_causal_model",
        lambda **_kwargs: (nn.Linear(1, 1), quantization),
    )

    def attach_adapter(base_model, config):
        attached.update(base_model=base_model, config=config)
        return adapter

    monkeypatch.setattr(trainer_module, "get_peft_model", attach_adapter)

    trainer = Trainer(
        model_or_id="base-model",
        schema=schema(answer=choice(["Yes", "No"])),
        config=TrainingConfig(device="mps", adaptation="qlora"),
    )

    assert trainer.model is adapter
    assert adapter.to_called is False
    assert attached["config"].target_modules == "all-linear"
    assert trainer.torch_dtype == torch.float16
    assert trainer.quantization_metadata == quantization


def test_existing_adapter_is_loaded_as_trainable(tmp_path, monkeypatch):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base-model"}),
        encoding="utf-8",
    )
    (adapter_path / "adapter_model.safetensors").write_bytes(b"")

    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token=None,
        eos_token="<eos>",
    )
    adapter = TrainableAdapter()
    loaded = {}

    monkeypatch.setattr(
        "jeeva.training.trainer.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        trainer_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(
        "jeeva.training.trainer.PeftConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_model_name_or_path="base-model",
            revision=None,
        ),
    )
    monkeypatch.setattr(
        "jeeva.training.trainer._load_causal_or_vision_model",
        lambda **_kwargs: nn.Linear(1, 1),
    )

    def load_adapter(base_model, path, *, is_trainable, **kwargs):
        loaded.update(
            base_model=base_model,
            path=path,
            is_trainable=is_trainable,
            kwargs=kwargs,
        )
        return adapter

    monkeypatch.setattr(
        "jeeva.training.trainer.PeftModel.from_pretrained",
        load_adapter,
    )

    trainer = Trainer(
        model_or_id=str(adapter_path),
        schema=schema(answer=choice(["Yes", "No"])),
        config=TrainingConfig(device="cpu"),
    )

    assert trainer.model is adapter
    assert loaded["path"] == str(adapter_path)
    assert loaded["is_trainable"] is True


def test_existing_adapter_continues_with_qlora_without_model_move(tmp_path, monkeypatch):
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"")
    tokenizer = SimpleNamespace(
        padding_side="right",
        pad_token=None,
        eos_token="<eos>",
    )
    adapter = QLoRAAdapter()
    loaded = {}
    quantization = {
        "backend": "bitsandbytes",
        "backend_version": "0.50.2",
        "bits": 4,
        "quant_type": "nf4",
        "double_quantization": True,
        "compute_dtype": "float16",
    }

    monkeypatch.setattr(
        trainer_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        trainer_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
    )
    monkeypatch.setattr(
        trainer_module.PeftConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            base_model_name_or_path="base-model",
            revision=None,
        ),
    )
    monkeypatch.setattr(
        trainer_module,
        "_load_qlora_causal_model",
        lambda **_kwargs: (nn.Linear(1, 1), quantization),
    )

    def load_adapter(base_model, path, *, is_trainable, **kwargs):
        loaded.update(
            base_model=base_model,
            path=path,
            is_trainable=is_trainable,
            kwargs=kwargs,
        )
        return adapter

    monkeypatch.setattr(trainer_module.PeftModel, "from_pretrained", load_adapter)

    trainer = Trainer(
        model_or_id=str(adapter_path),
        schema=schema(answer=choice(["Yes", "No"])),
        config=TrainingConfig(device="mps", adaptation="qlora"),
    )

    assert trainer.model is adapter
    assert adapter.to_called is False
    assert loaded["path"] == str(adapter_path)
    assert loaded["is_trainable"] is True
    assert trainer.quantization_metadata == quantization
