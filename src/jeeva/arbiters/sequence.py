from typing import Any

import torch

from jeeva.arbiters.base import (
    BaseArbiter,
    load_arbiter_model,
    prefill_context,
)
from jeeva.core.cache import (
    build_branch_attention_and_positions,
    replicate_dynamic_cache,
)
from jeeva.schema.primitives import BasePrimitive, DecisionResult


class SequenceArbiter(BaseArbiter):
    """
    Decoder Sequence Log-Likelihood Arbiter.
    Evaluates ambiguous candidate options by reusing the gatekeeper's prefilled document
    KV-cache with the same decoder model. Prefills the field query prefix once into a cache
    clone, then scores all candidate option phrases in a single batched forward pass by
    computing their exact length-normalized causal sequence log-likelihood:
        score(o_k) = (1/T_k) * sum_t log P(w_t | doc, prefix, w_<t)

    Zero tokens generated. No separate model required.
    Executes in ~10-15ms.

    Supports dual mode:
    - Default (model_or_id=None): Reuses the Gatekeeper's model, tokenizer, and KV-cache.
    - Custom model (model_or_id="..."): Operates on a dedicated, larger, or fine-tuned model.
    """

    def __init__(
        self,
        model_or_id: str | Any | None = None,
        device: str | None = None,
        dtype: torch.dtype | str | Any = None,
        torch_dtype: torch.dtype | str | Any = None,
        ambiguity_criterion: str = "margin",
        contiguous_cache: bool | None = None,
    ):
        self.model_or_id = model_or_id
        self.device = device
        self.torch_dtype = dtype if dtype is not None else torch_dtype
        self.ambiguity_criterion = ambiguity_criterion.lower()
        self.contiguous_cache = contiguous_cache
        self._model = None
        self._tokenizer = None
        self._target_device = None

    @property
    def name(self) -> str:
        return "SequenceArbiter"

    @property
    def uses_primary_model(self) -> bool:
        return self.model_or_id is None

    def resolve(
        self,
        context: str,
        ambiguous_fields: dict[str, BasePrimitive],
        current_decisions: dict[str, DecisionResult],
        engine: Any,
        base_cache: Any,
        past_len: int,
        threshold: float,
    ) -> dict[str, DecisionResult]:
        resolved: dict[str, DecisionResult] = {}

        if self.model_or_id is not None:
            if self._model is None:
                self._model, self._tokenizer, self._target_device = load_arbiter_model(
                    self.model_or_id, self.device, self.torch_dtype
                )
            model, tokenizer, target_device = (
                self._model,
                self._tokenizer,
                str(self._target_device or "cpu"),
            )
            active_cache, active_past_len, branch_suffix = prefill_context(
                model, tokenizer, context, target_device
            )
        else:
            if engine is None:
                raise ValueError("SequenceArbiter requires a dedicated causal model.")
            model, tokenizer, target_device = (
                engine.model,
                engine.tokenizer,
                str(engine.device or "cpu"),
            )
            active_cache, active_past_len = base_cache, past_len
            branch_suffix = engine._branch_suffix

        if tokenizer is None:
            raise ValueError("Engine tokenizer is uninitialized.")
        use_contiguous_cache = (
            (engine.contiguous_cache if engine is not None else True)
            if self.contiguous_cache is None
            else self.contiguous_cache
        )
        eos = tokenizer.eos_token or ""

        for field_key, primitive in ambiguous_fields.items():
            options = primitive.options
            prefix = f"{branch_suffix}{primitive.build_branch_prompt()}"
            cand_continuations = [f"{option}.{eos}" for option in options]

            num_candidates = len(options)

            # Step 1: Prefill the field query prefix ONCE into a single clone of the document KV cache
            prefix_inputs = tokenizer(prefix, return_tensors="pt").to(target_device)
            prefix_cache = replicate_dynamic_cache(
                active_cache,
                num_lanes=1,
                contiguous=use_contiguous_cache,
            )
            comb_prefix_mask, prefix_pos_ids = build_branch_attention_and_positions(
                past_len=active_past_len,
                branch_attention_mask=prefix_inputs.attention_mask,
                device=torch.device(target_device),
            )

            with torch.no_grad():
                prefix_out = model(
                    input_ids=prefix_inputs.input_ids,
                    attention_mask=comb_prefix_mask,
                    position_ids=prefix_pos_ids,
                    past_key_values=prefix_cache,
                    use_cache=True,
                )

            prefix_past_len = prefix_cache.get_seq_length()
            # The last token of prefix produces the logit distribution predicting the 1st option token
            prefix_last_logit = prefix_out.logits[:, -1:, :]

            # Step 2: Forward-pass ONLY candidate continuation tokens over replicated prefix cache
            cand_inputs = tokenizer(cand_continuations, return_tensors="pt", padding=True).to(
                target_device
            )
            expanded_cache = replicate_dynamic_cache(
                prefix_cache,
                num_lanes=num_candidates,
                contiguous=use_contiguous_cache,
            )

            comb_mask, pos_ids = build_branch_attention_and_positions(
                past_len=prefix_past_len,
                branch_attention_mask=cand_inputs.attention_mask,
                device=torch.device(target_device),
            )

            with torch.no_grad():
                cand_out = model(
                    input_ids=cand_inputs.input_ids,
                    attention_mask=comb_mask,
                    position_ids=pos_ids,
                    past_key_values=expanded_cache,
                    use_cache=False,
                )

            # Step 3: Combine prefix prediction with candidate sequence logits
            opt_logits = torch.cat(
                [prefix_last_logit.expand(num_candidates, 1, -1), cand_out.logits[:, :-1, :]],
                dim=1,
            )
            opt_log_probs = torch.log_softmax(opt_logits, dim=-1)
            opt_targets = cand_inputs.input_ids.unsqueeze(-1)
            token_log_probs = opt_log_probs.gather(dim=-1, index=opt_targets).squeeze(-1)

            scoring_mask = cand_inputs.attention_mask.clone()
            token_counts = scoring_mask.sum(dim=-1).clamp(min=1)
            seq_log_likelihood = (token_log_probs * scoring_mask).sum(dim=-1) / token_counts

            probs = torch.softmax(seq_log_likelihood, dim=-1).detach().to("cpu", torch.float32)
            result = primitive.result_from_probabilities(
                probs,
                cascade_threshold=threshold,
                resolved_by=self.name,
            )
            certainty_score = (
                result.margin if result.certainty_score is None else result.certainty_score
            )
            if self.ambiguity_criterion == "entropy":
                result.is_certain = bool(
                    result.normalized_entropy is not None
                    and result.normalized_entropy <= (1.0 - result.tau)
                )
            elif self.ambiguity_criterion == "both":
                result.is_certain = bool(
                    certainty_score >= result.tau
                    and result.normalized_entropy is not None
                    and result.normalized_entropy <= (1.0 - result.tau)
                )
            resolved[field_key] = result

        return resolved
