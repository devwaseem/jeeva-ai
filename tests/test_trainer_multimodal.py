from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from jeeva.data.dataset import JeevaDataset
from jeeva.schema.multimodal import MultimodalContext
from jeeva.training.trainer import Trainer, TrainingConfig
from tests.schema_helpers import choice, noul, schema


def test_jeeva_dataset_with_single_image():
    decision_schema = schema(category=choice(["Invoice", "Receipt"]))
    records = [
        {"text": "Invoice from vendor", "image": "invoice.png", "category": "Invoice"},
    ]
    ds = JeevaDataset(data=records, schema=decision_schema)
    assert len(ds) == 1
    sample = ds[0]
    doc = sample["document"]
    assert isinstance(doc, MultimodalContext)
    assert doc.text == "Invoice from vendor"
    assert doc.images == ["invoice.png"]
    assert doc.has_images() is True


def test_jeeva_dataset_with_multiple_images():
    decision_schema = schema(signed=noul("The document is signed."))
    records = [
        {"context": "Contract pages", "images": ["p1.png", "p2.png"], "signed": True},
    ]
    ds = JeevaDataset(data=records, schema=decision_schema)
    assert len(ds) == 1
    sample = ds[0]
    doc = sample["document"]
    assert isinstance(doc, MultimodalContext)
    assert len(doc.images) == 2
    assert sample["lane_mask"][0] == 1.0


def test_jeeva_dataset_with_image_only_uses_default_prompt():
    decision_schema = schema(category=choice(["A", "B"]))
    records = [
        {"image": "photo.jpg", "category": "A"},
    ]
    ds = JeevaDataset(data=records, schema=decision_schema)
    sample = ds[0]
    doc = sample["document"]
    assert isinstance(doc, MultimodalContext)
    assert doc.text == "Analyze the provided image."
    assert doc.images == ["photo.jpg"]


def test_jeeva_dataset_backward_compatible_text_only():
    decision_schema = schema(category=choice(["Billing", "Tech"]))
    records = [
        {"document": "My server crashed", "category": "Tech"},
    ]
    ds = JeevaDataset(data=records, schema=decision_schema)
    sample = ds[0]
    # Backward compatible: returns plain str when no images are present
    assert isinstance(sample["document"], str)
    assert sample["document"] == "My server crashed"


def test_trainer_build_prefill_inputs_multimodal():
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.is_multimodal = True

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.side_effect = lambda messages, **_: (
        f"<formatted_prompt>{messages[1]['content'][-1]['text'].splitlines()[-1]}<assistant>"
    )
    t = torch.tensor([[1, 2, 3]])
    mock_processor.return_value = {"input_ids": t, "pixel_values": t}
    trainer.processor = mock_processor

    ctx = MultimodalContext(text="Audit receipt", images=["receipt.jpg"])
    mock_img = MagicMock()
    with patch.object(ctx, "load_pil_images", return_value=[mock_img]):
        inputs = trainer._build_prefill_inputs(ctx)

    assert "input_ids" in inputs
    assert "pixel_values" in inputs
    mock_processor.apply_chat_template.assert_called_once()
    mock_processor.assert_called_once_with(
        text="<formatted_prompt>",
        images=[mock_img],
        return_tensors="pt",
    )


def test_trainer_text_only_rejects_images_with_error():
    trainer = object.__new__(Trainer)
    trainer.device = "cpu"
    trainer.is_multimodal = False
    trainer.processor = None

    ctx = MultimodalContext(text="Analyze", images=["img.png"])
    import pytest

    with pytest.raises(ValueError, match="requires a vision-language model"):
        trainer._build_prefill_inputs(ctx)


def test_trainer_init_vlm_lora_task_type():
    mock_base_model = MagicMock()
    mock_processor = MagicMock()
    mock_processor.image_processor = MagicMock()
    mock_processor.tokenizer = MagicMock()

    mock_peft_model = MagicMock()
    mock_peft_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]

    with (
        patch(
            "jeeva.training.trainer._load_causal_or_vision_model",
            return_value=mock_base_model,
        ),
        patch(
            "jeeva.training.trainer.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(vision_config=object()),
        ),
        patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor),
        patch(
            "jeeva.training.trainer.get_peft_model", return_value=mock_peft_model
        ) as mock_get_peft,
    ):
        decision_schema = schema(cat=choice(["A", "B"]))
        cfg = TrainingConfig(epochs=1)
        trainer = Trainer(
            model_or_id="Qwen/Qwen2-VL-2B-Instruct",
            schema=decision_schema,
            config=cfg,
        )

        assert trainer.is_multimodal is True
        assert trainer.processor == mock_processor
        # Verify LoRA task_type was set to None for multimodal architecture
        mock_get_peft.assert_called_once()
        lora_cfg = mock_get_peft.call_args[0][1]
        assert lora_cfg.task_type is None
