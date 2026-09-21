import copy
from types import MethodType
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


def replicate_dynamic_cache(
    base_cache: DynamicCache,
    num_lanes: int,
    *,
    contiguous: bool = True,
    autograd_safe: bool = False,
) -> DynamicCache:
    """Replicate a document KV cache across branch lanes.

    Contiguous replication is the default because attention kernels generally
    achieve lower latency on dense tensors. Setting ``contiguous=False`` uses a
    zero-copy expanded view for immutable attention KV tensors when the base
    cache batch size is one. Model-specific states updated in place remain dense.
    Set ``autograd_safe=True`` when gradients flow through the cache so linear
    attention state updates do not invalidate tensors saved for backward.
    """
    if num_lanes <= 0:
        raise ValueError(f"num_lanes must be >= 1, got {num_lanes}")

    def replicate(tensor: torch.Tensor) -> torch.Tensor:
        if contiguous:
            return tensor.repeat_interleave(num_lanes, dim=0).contiguous()
        if tensor.shape[0] != 1:
            raise ValueError(
                "Non-contiguous cache expansion requires a base cache batch size of one."
            )
        return tensor.expand(num_lanes, *tensor.shape[1:])

    def replicate_mutable(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat_interleave(num_lanes, dim=0).contiguous()

    def replace_recurrent_state(
        layer: Any,
        recurrent_states: torch.Tensor,
        state_idx: int = 0,
        **_: Any,
    ) -> torch.Tensor:
        initialized = layer.is_recurrent_states_initialized
        if not initialized[state_idx]:
            layer.lazy_initialization(
                recurrent_states=recurrent_states,
                state_idx=state_idx,
            )
        layer.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    expanded = copy.copy(base_cache)
    expanded_any: Any = expanded

    # Transformers 5.x stores model-specific cache layers in ``.layers``.
    # Preserve those layer types: hybrid models such as Qwen3.5 interleave
    # DynamicLayer and LinearAttentionLayer instances.
    layers: Any = getattr(base_cache, "layers", None)
    if layers:
        expanded_any.layers = []
        for layer in layers:
            expanded_layer = copy.copy(layer)

            # Linear-attention layers keep state and initialization metadata in
            # dictionaries. Copy every dictionary so later model updates cannot
            # mutate the base cache through the replicated layer.
            for name, value in vars(layer).items():
                if isinstance(value, dict):
                    setattr(expanded_layer, name, value.copy())

            if autograd_safe and hasattr(expanded_layer, "recurrent_states"):
                # Transformers updates recurrent state with copy_(), while Qwen3.5
                # has already saved the prior state for backward. Use out-of-place
                # state replacement and the layer's native out-of-place conv path.
                expanded_layer.record_past = True
                expanded_layer.update_recurrent_state = MethodType(
                    replace_recurrent_state,
                    expanded_layer,
                )

            initialized = False
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None or values is not None:
                if keys is None or values is None:
                    raise ValueError("Base cache contains a partially initialized attention layer.")
                expanded_layer.keys = replicate(keys)
                expanded_layer.values = replicate(values)
                initialized = True

            for state_name in ("conv_states", "recurrent_states"):
                states = getattr(layer, state_name, None)
                if states is None:
                    continue
                expanded_states = getattr(expanded_layer, state_name)
                for state_idx, state in states.items():
                    if state is not None:
                        expanded_states[state_idx] = replicate_mutable(state)
                        initialized = True

            if not initialized:
                raise ValueError("Base cache contains an uninitialized layer.")
            expanded_any.layers.append(expanded_layer)
        return expanded

    # Fallback for earlier DynamicCache implementations.
    key_cache: Any = getattr(base_cache, "key_cache", None)
    value_cache: Any = getattr(base_cache, "value_cache", None)
    if key_cache:
        if value_cache is None:
            raise ValueError("Base cache contains keys without values.")
        expanded_vars = vars(expanded)
        expanded_vars["key_cache"] = [replicate(key) for key in key_cache]
        expanded_vars["value_cache"] = [replicate(value) for value in value_cache]
        return expanded

    raise ValueError("Base cache contains no initialized layers to replicate.")


def build_branch_attention_and_positions(
    past_len: int,
    branch_attention_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the combined left-padded attention mask and continuous position IDs
    for parallel context slicing over a shared KV cache.
    """
    num_lanes, branch_seq_len = branch_attention_mask.shape

    # Construct past mask for base document (all active tokens)
    past_mask = torch.ones(
        (num_lanes, past_len),
        dtype=branch_attention_mask.dtype,
        device=device,
    )

    # Combined mask: [num_lanes, past_len + branch_seq_len]
    comb_mask = torch.cat([past_mask, branch_attention_mask], dim=1).contiguous()

    # Compute position IDs aligned to the continuous sequence
    pos_ids = (comb_mask.cumsum(dim=-1)[:, -branch_seq_len:] - 1).contiguous()
    pos_ids.masked_fill_(branch_attention_mask == 0, 0)

    return comb_mask, pos_ids
