from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jeeva.core.engine import DecisionEngine
from jeeva.schema.multimodal import MultimodalContext


def test_multimodal_context_validation_empty_text():
    with pytest.raises(ValueError, match="non-empty string"):
        MultimodalContext(text="")

    with pytest.raises(ValueError, match="non-empty string"):
        MultimodalContext(text="   \n\t  ")


def test_multimodal_context_audio_not_implemented():
    with pytest.raises(NotImplementedError, match="Audio inputs are not yet supported"):
        MultimodalContext(text="Hello", audio=["sample.mp3"])


def test_multimodal_context_invalid_image_type():
    with pytest.raises(ValueError, match="images\\[0\\] must be a local file path"):
        MultimodalContext(text="Hello", images=[12345])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="images\\[1\\] must be a local file path"):
        MultimodalContext(text="Hello", images=["valid.png", {"not": "an_image"}])  # type: ignore[list-item]


def test_multimodal_context_valid_construction(tmp_path: Path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake-image-bytes")

    ctx = MultimodalContext(
        text="Classify this document",
        images=[str(img_file), img_file],
    )
    assert ctx.text == "Classify this document"
    assert len(ctx.images) == 2
    assert ctx.has_images() is True
    assert "MultimodalContext" in repr(ctx)
    assert "2 image(s)" in repr(ctx)


def test_multimodal_context_no_images():
    ctx = MultimodalContext(text="Just text context")
    assert ctx.has_images() is False
    assert len(ctx.images) == 0


def test_multimodal_context_load_missing_file(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.png"
    ctx = MultimodalContext(text="Test", images=[missing_file])
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        ctx.load_pil_images()


def test_multimodal_context_load_real_image(tmp_path: Path):
    from PIL import Image

    img_path = tmp_path / "sample.png"
    img = Image.new("RGB", (32, 32), color="red")
    img.save(img_path)

    ctx = MultimodalContext(text="Classify image", images=[img_path])
    loaded = ctx.load_pil_images()
    assert len(loaded) == 1
    assert loaded[0].size == (32, 32)


def test_multimodal_context_with_pil_instance():
    from PIL import Image

    img = Image.new("RGB", (16, 16), color="blue")
    ctx = MultimodalContext(text="Classify image", images=[img])
    loaded = ctx.load_pil_images()
    assert len(loaded) == 1
    assert loaded[0] is img


def test_multimodal_context_max_image_size(tmp_path: Path):
    from PIL import Image

    img_path = tmp_path / "large.png"
    img = Image.new("RGB", (1000, 500), color="blue")
    img.save(img_path)

    ctx = MultimodalContext(text="Test", images=[img_path], max_image_size=400)
    loaded = ctx.load_pil_images()
    assert len(loaded) == 1
    assert max(loaded[0].size) <= 400


def test_multimodal_context_pillow_missing():
    ctx = MultimodalContext(text="Test", images=["some_image.png"])
    with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
        with pytest.raises(ImportError, match="Pillow is required"):
            ctx.load_pil_images()


def test_multimodal_context_rejects_remote_images():
    with pytest.raises(ValueError, match="Remote image URLs are not supported"):
        MultimodalContext(text="Test", images=["https://example.com/image.png"])


def test_multimodal_context_bounds_image_count():
    with pytest.raises(ValueError, match="At most 1 images"):
        MultimodalContext(text="Test", images=["one.png", "two.png"], max_images=1)


def test_multimodal_context_bounds_decoded_image_bytes(tmp_path: Path):
    from PIL import Image

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (10, 10), color="red").save(first)
    Image.new("RGB", (10, 10), color="blue").save(second)

    with pytest.raises(ValueError, match="per-image limit"):
        MultimodalContext(
            text="Test",
            images=[first],
            max_decoded_bytes_per_image=299,
        ).load_pil_images()

    with pytest.raises(ValueError, match="aggregate"):
        MultimodalContext(
            text="Test",
            images=[first, second],
            max_total_decoded_bytes=599,
        ).load_pil_images()


def test_import_without_pillow():
    import importlib

    with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
        import jeeva.schema.multimodal

        importlib.reload(jeeva.schema.multimodal)
        assert hasattr(jeeva.schema.multimodal, "MultimodalContext")
    # Reload again with normal sys.modules
    importlib.reload(jeeva.schema.multimodal)


def test_engine_text_only_rejects_images():
    # Construct a mock DecisionEngine without running HF from_pretrained
    engine = object.__new__(DecisionEngine)
    engine.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    engine.device = "cpu"
    engine.is_multimodal = False
    engine.processor = None

    ctx = MultimodalContext(text="Examine this image", images=["doc.png"])
    with pytest.raises(ValueError, match="requires a vision-language model"):
        engine._build_prefill_inputs(ctx)


def test_engine_extract_context_text():
    engine = object.__new__(DecisionEngine)
    # str context
    assert engine._extract_context_text("raw text") == "raw text"
    # MultimodalContext
    ctx = MultimodalContext(text="extracted text")
    assert engine._extract_context_text(ctx) == "extracted text"


def test_engine_build_multimodal_message():
    engine = object.__new__(DecisionEngine)
    ctx = MultimodalContext(text="Analyze invoice", images=["img1.png", "img2.png"])
    parts = engine._build_multimodal_message(ctx)

    assert len(parts) == 3
    assert parts[0] == {"type": "image"}
    assert parts[1] == {"type": "image"}
    assert parts[2] == {"type": "text", "text": "Document:\nAnalyze invoice"}


def test_engine_text_only_path_with_multimodal_context_without_images():
    engine = object.__new__(DecisionEngine)
    engine.device = "cpu"
    engine.is_multimodal = False
    engine.tokenizer = MagicMock()
    mock_tensor_dict = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer_out = MagicMock()
    mock_tokenizer_out.to.return_value = mock_tensor_dict
    engine.tokenizer.return_value = mock_tokenizer_out

    ctx = MultimodalContext(text="Document without images")
    inputs = engine._build_prefill_inputs(ctx)

    assert inputs == mock_tensor_dict
    engine.tokenizer.assert_called_once()
    prompt_arg = engine.tokenizer.call_args[0][0]
    assert "Document:\nDocument without images" in prompt_arg


def test_engine_multimodal_prefill_with_mock_processor():
    engine = object.__new__(DecisionEngine)
    engine.device = "cpu"
    engine.is_multimodal = True
    engine.model_id = "Qwen/Qwen2-VL-2B-Instruct"

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.side_effect = lambda messages, **_: (
        f"<mock_prompt>{messages[1]['content'][-1]['text'].splitlines()[-1]}<assistant>"
    )

    # Mock tensor output
    t = torch.tensor([[1, 2]])
    mock_processor.return_value = {"input_ids": t, "pixel_values": t}
    engine.processor = mock_processor

    ctx = MultimodalContext(text="Analyze chart", images=["chart.png"])
    # Mock load_pil_images
    mock_img = MagicMock()
    with patch.object(ctx, "load_pil_images", return_value=[mock_img]):
        inputs = engine._build_prefill_inputs(ctx)

    assert "input_ids" in inputs
    assert "pixel_values" in inputs
    mock_processor.apply_chat_template.assert_called_once()
    mock_processor.assert_called_once_with(
        text="<mock_prompt>",
        images=[mock_img],
        return_tensors="pt",
    )


def test_load_causal_or_vision_model_vlm_fallback():
    from jeeva.core.engine import _load_causal_or_vision_model

    mock_causal = MagicMock()
    mock_causal.from_pretrained.side_effect = ValueError(
        "Unrecognized configuration class for this kind of AutoModel: AutoModelForCausalLM"
    )

    mock_vlm_model = MagicMock()
    mock_vlm_loader = MagicMock()
    mock_vlm_loader.from_pretrained.return_value = mock_vlm_model

    with (
        patch("jeeva.core.engine.AutoModelForCausalLM", mock_causal),
        patch("transformers.AutoModelForImageTextToText", mock_vlm_loader, create=True),
    ):
        model = _load_causal_or_vision_model("Qwen/Qwen2-VL-2B-Instruct", "cpu", torch.float32)
        assert model == mock_vlm_model.to("cpu")
        mock_causal.from_pretrained.assert_called_once()
        mock_vlm_loader.from_pretrained.assert_called_once()


def test_get_letter_weight_with_get_output_embeddings():
    engine = object.__new__(DecisionEngine)
    engine._letter_weights = {}
    engine.device = "cpu"

    engine.tokenizer = MagicMock()
    engine.tokenizer.encode.return_value = [42]

    # Model has get_output_embeddings() instead of lm_head attribute
    mock_linear = MagicMock()
    weight_tensor = torch.randn(100, 16)
    mock_linear.weight = weight_tensor

    mock_model = MagicMock(spec=[])
    mock_model.get_output_embeddings = MagicMock(return_value=mock_linear)
    engine.model = mock_model

    weight = engine._get_letter_weight("A")
    assert torch.equal(weight, weight_tensor[42].to(torch.float32))
    mock_model.get_output_embeddings.assert_called_once()
