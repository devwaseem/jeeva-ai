import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from jeeva.core.cache import (
    build_branch_attention_and_positions,
    replicate_dynamic_cache,
)


def test_replicate_dynamic_cache():
    # Construct a synthetic DynamicCache with 2 layers
    base_cache = DynamicCache()
    k1 = torch.randn(1, 4, 10, 32)
    v1 = torch.randn(1, 4, 10, 32)
    base_cache.update(k1, v1, layer_idx=0)

    k2 = torch.randn(1, 4, 10, 32)
    v2 = torch.randn(1, 4, 10, 32)
    base_cache.update(k2, v2, layer_idx=1)

    # Replicate across 3 lanes
    num_lanes = 3
    expanded = replicate_dynamic_cache(base_cache, num_lanes=num_lanes)

    assert len(expanded.layers) == 2
    for layer in expanded.layers:
        assert layer.keys.shape[0] == num_lanes
        assert layer.values.shape[0] == num_lanes
        assert layer.keys.shape[2] == 10  # seq_len preserved

    # Verify base cache was not mutated
    assert base_cache.layers[0].keys.shape[0] == 1


def test_replicate_dynamic_cache_can_use_zero_copy_expanded_views():
    base_cache = DynamicCache()
    keys = torch.randn(1, 2, 4, 8)
    values = torch.randn(1, 2, 4, 8)
    base_cache.update(keys, values, layer_idx=0)

    expanded = replicate_dynamic_cache(base_cache, num_lanes=3, contiguous=False)
    layer = expanded.layers[0]

    assert layer.keys.shape == (3, 2, 4, 8)
    assert layer.values.shape == (3, 2, 4, 8)
    assert layer.keys.is_contiguous() is False
    assert layer.values.is_contiguous() is False
    base_layer = base_cache.layers[0]
    assert layer.keys.untyped_storage().data_ptr() == base_layer.keys.untyped_storage().data_ptr()
    assert (
        layer.values.untyped_storage().data_ptr() == base_layer.values.untyped_storage().data_ptr()
    )


def test_replicate_dynamic_cache_supports_linear_attention_layers():
    base_cache = DynamicCache()
    linear_layer = LinearAttentionLayer()
    conv_state = torch.randn(1, 4, 3)
    recurrent_state = torch.randn(1, 4, 8)
    linear_layer.update_conv_state(conv_state, conv_kernel_size=3)
    linear_layer.update_recurrent_state(recurrent_state)

    dynamic_layer = DynamicLayer()
    keys = torch.randn(1, 2, 4, 8)
    values = torch.randn(1, 2, 4, 8)
    dynamic_layer.update(keys, values)
    base_cache.layers = [linear_layer, dynamic_layer]

    for contiguous in (True, False):
        expanded = replicate_dynamic_cache(
            base_cache,
            num_lanes=3,
            contiguous=contiguous,
        )

        expanded_linear = expanded.layers[0]
        assert isinstance(expanded_linear, LinearAttentionLayer)
        assert torch.equal(
            expanded_linear.conv_states[0],
            conv_state.expand(3, -1, -1),
        )
        assert torch.equal(
            expanded_linear.recurrent_states[0],
            recurrent_state.expand(3, -1, -1),
        )
        assert expanded.layers[1].keys.shape == (3, 2, 4, 8)
        assert expanded_linear.conv_states is not linear_layer.conv_states
        assert expanded_linear.has_previous_state is not linear_layer.has_previous_state
        assert expanded_linear.record_past is False
        assert (
            expanded_linear.conv_states[0].untyped_storage().data_ptr()
            != linear_layer.conv_states[0].untyped_storage().data_ptr()
        )

        next_conv_state = torch.randn(3, 4, 1)
        next_recurrent_state = torch.randn(3, 4, 8)
        expanded_linear.update_conv_state(next_conv_state)
        expanded_linear.update_recurrent_state(next_recurrent_state)
        assert torch.equal(expanded_linear.conv_states[0][..., -1], next_conv_state[..., 0])
        assert torch.equal(expanded_linear.recurrent_states[0], next_recurrent_state)

    assert torch.equal(base_cache.layers[0].conv_states[0], conv_state)
    assert torch.equal(base_cache.layers[0].recurrent_states[0], recurrent_state)


@pytest.mark.training_integration
def test_qwen35_hybrid_cache_supports_backward_through_prefill():
    qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    config = qwen3_5.Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )
    model = AutoModelForCausalLM.from_config(config)
    base_out = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
    expanded_cache = replicate_dynamic_cache(
        base_out.past_key_values,
        num_lanes=2,
        autograd_safe=True,
    )
    branch_mask = torch.ones((2, 2), dtype=torch.long)
    attention_mask, position_ids = build_branch_attention_and_positions(
        past_len=4,
        branch_attention_mask=branch_mask,
        device=torch.device("cpu"),
    )

    branch_out = model(
        input_ids=torch.tensor([[5, 6], [7, 8]]),
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=expanded_cache,
        use_cache=False,
        output_hidden_states=True,
    )
    branch_out.hidden_states[-1][:, -1, :].square().mean().backward()

    assert model.model.embed_tokens.weight.grad is not None


def test_build_branch_attention_and_positions():
    past_len = 5
    # Lane 0 has 1 pad token on the left, Lane 1 has 0 pad tokens
    branch_mask = torch.tensor(
        [
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.long,
    )

    comb_mask, pos_ids = build_branch_attention_and_positions(
        past_len=past_len,
        branch_attention_mask=branch_mask,
        device=torch.device("cpu"),
    )

    # comb_mask shape should be [2, 5 + 3] = [2, 8]
    assert comb_mask.shape == (2, 8)
    # First 5 positions in comb_mask are all 1s
    assert (comb_mask[:, :past_len] == 1).all()

    # Position IDs for active tokens in lane 0 should follow past_len
    # cumsum for lane 0: [1, 2, 3, 4, 5, 5, 6, 7] - 1 = [0, 1, 2, 3, 4, 4, 5, 6]
    # last 3: [4, 5, 6], with masked fill for pad: [0, 5, 6]
    assert pos_ids[0].tolist() == [0, 5, 6]
    # lane 1: [5, 6, 7]
    assert pos_ids[1].tolist() == [5, 6, 7]
