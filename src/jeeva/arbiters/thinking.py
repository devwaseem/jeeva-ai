from typing import Any, Literal

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
from jeeva.core.candidate_scoring import (
    CandidateScoreAggregate,
    aggregate_candidate_scores,
    balanced_candidate_orders,
)
from jeeva.schema.primitives import BasePrimitive, DecisionResult


def _ends_with_token_sequence(tokens: list[int], suffix: list[int]) -> bool:
    return bool(suffix) and len(tokens) >= len(suffix) and tokens[-len(suffix) :] == suffix


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled)

    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _build_thinking_prompt(
    *,
    branch_suffix: str,
    prompt_label: str,
    options: list[Any],
) -> str:
    options_str = ", ".join(f"'{option}'" for option in options)
    return (
        f"{branch_suffix}<think>\n"
        "Analyze this finite decision field using only evidence in the document.\n"
        f"{prompt_label}\n"
        f"Candidate choices: {options_str}\n"
        "Instructions:\n"
        "1. Treat the field definition as authoritative and exhaustive; do not add unstated requirements.\n"
        "2. Identify the exact document evidence relevant to this field.\n"
        "3. Resolve entities, pronouns, and who each fact applies to.\n"
        "4. Check negation, hypotheticals, general facts, and historical references.\n"
        "5. Compare the evidence for and against every candidate.\n"
        "6. Do not state a final verdict, conclusion, answer, or selected candidate.\n"
        "End after the evidence analysis, then </think>.\n"
        "Evidence analysis:"
    )


class SequenceThinkingArbiter(BaseArbiter):
    """
    Autoregressive Chain-of-Thought Thinking Arbiter.

    For each field selected by the engine, this arbiter:
    1. Reuses the prefilled document KV-cache (or prefills using its own dedicated model).
    2. Autoregressively generates a verdict-free evidence trace inside ``<think>``.
    3. Evaluates the finite candidates using single-token letter logits by default, or
       optional length-normalized sequence likelihood.
    4. Attaches the evidence trace directly to ``DecisionResult.reasoning`` for
       auditability.

    Supports dual mode:
    - Default (model_or_id=None): Inherits model/tokenizer from Gatekeeper with zero extra prefill.
    - Dedicated model (model_or_id="..."): Runs on a separate/larger model (e.g. 1.5B/7B/QwQ).
    """

    def __init__(
        self,
        model_or_id: str | Any | None = None,
        device: str | None = None,
        dtype: torch.dtype | str | Any = None,
        torch_dtype: torch.dtype | str | Any = None,
        max_thinking_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
        ambiguity_criterion: str = "margin",
        contiguous_cache: bool | None = None,
        always_think: bool = False,
        readout: Literal["letter", "sequence"] = "letter",
        candidate_position_balancing: bool = True,
    ):
        """
        Args:
            model_or_id: Optional separate model ID or instance for thinking.
            device: Optional target device for custom model.
            dtype: Optional torch dtype or string ('float16', 'bfloat16', 'float32') for custom model.
            torch_dtype: Alias for dtype.
            max_thinking_tokens: Maximum number of tokens the model is allowed to think.
            temperature: Sampling temperature for reasoning generation (0.0 for greedy).
            top_p: Nucleus sampling probability threshold.
            ambiguity_criterion: Ambiguity evaluation metric ("margin", "entropy", or "both").
            always_think: Resolve every field instead of only uncertain gatekeeper fields.
            readout: Candidate scoring strategy after reasoning ("letter" or "sequence").
            candidate_position_balancing: Evaluate balanced letter orders before pooling scores.
        """
        self.model_or_id = model_or_id
        self.device = device
        self.torch_dtype = dtype if dtype is not None else torch_dtype
        if max_thinking_tokens < 1:
            raise ValueError("max_thinking_tokens must be at least 1.")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        criterion = ambiguity_criterion.lower()
        if criterion not in ("margin", "entropy", "both"):
            raise ValueError("ambiguity_criterion must be 'margin', 'entropy', or 'both'.")
        if readout not in ("letter", "sequence"):
            raise ValueError("readout must be 'letter' or 'sequence'.")
        self.max_thinking_tokens = max_thinking_tokens
        self.temperature = max(0.0, temperature)
        self.top_p = top_p
        self.ambiguity_criterion = criterion
        self.contiguous_cache = contiguous_cache
        self.always_think = always_think
        self.readout = readout
        self.candidate_position_balancing = candidate_position_balancing
        self._model = None
        self._tokenizer = None
        self._target_device = None

    @property
    def name(self) -> str:
        return "SequenceThinkingArbiter"

    @property
    def resolve_all_fields(self) -> bool:
        return self.always_think

    @property
    def uses_primary_model(self) -> bool:
        return self.model_or_id is None

    def _score_letter_candidates(
        self,
        *,
        model: Any,
        tokenizer: Any,
        target_device: str,
        think_cache: Any,
        options: list[Any],
        use_contiguous_cache: bool,
    ) -> CandidateScoreAggregate:
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if len(options) > len(letters):
            raise ValueError(
                f"Letter readout supports at most {len(letters)} candidates; "
                f"received {len(options)}. Use readout='sequence' instead."
            )
        candidate_letters = letters[: len(options)]
        orders = (
            balanced_candidate_orders(len(options))
            if self.candidate_position_balancing
            else (tuple(range(len(options))),)
        )
        answer_prompts = []
        for order in orders:
            ordered_options = [options[index] for index in order]
            option_lines = "\n".join(
                f"{letter}. {option}" for letter, option in zip(candidate_letters, ordered_options)
            )
            answer_prompts.append(f"\nOptions:\n{option_lines}\nAnswer: ")

        answer_inputs = tokenizer(
            answer_prompts,
            return_tensors="pt",
            padding=True,
        ).to(target_device)
        answer_cache = replicate_dynamic_cache(
            think_cache,
            num_lanes=len(orders),
            contiguous=use_contiguous_cache,
        )
        answer_mask, answer_positions = build_branch_attention_and_positions(
            past_len=think_cache.get_seq_length(),
            branch_attention_mask=answer_inputs.attention_mask,
            device=torch.device(target_device),
        )

        with torch.no_grad():
            answer_out = model(
                input_ids=answer_inputs.input_ids,
                attention_mask=answer_mask,
                position_ids=answer_positions,
                past_key_values=answer_cache,
                use_cache=False,
            )

        token_ids: list[int] = []
        for letter in candidate_letters:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"Letter candidate {letter!r} is not a single token for the "
                    "thinking arbiter model. Use readout='sequence' instead."
                )
            token_ids.append(encoded[0])

        next_token_logits = answer_out.logits[:, -1, :]
        candidate_ids = torch.tensor(token_ids, device=next_token_logits.device)
        positional_scores = next_token_logits.index_select(1, candidate_ids).to(torch.float32)
        return aggregate_candidate_scores(
            positional_scores,
            orders,
            temperature=1.0,
        )

    def _score_sequence_candidates(
        self,
        *,
        model: Any,
        tokenizer: Any,
        target_device: str,
        think_cache: Any,
        options: list[Any],
        use_contiguous_cache: bool,
    ) -> torch.Tensor:
        conclusion_inputs = tokenizer("\nConclusion: ", return_tensors="pt").to(target_device)
        conclusion_mask, conclusion_positions = build_branch_attention_and_positions(
            past_len=think_cache.get_seq_length(),
            branch_attention_mask=conclusion_inputs.attention_mask,
            device=torch.device(target_device),
        )

        with torch.no_grad():
            conclusion_out = model(
                input_ids=conclusion_inputs.input_ids,
                attention_mask=conclusion_mask,
                position_ids=conclusion_positions,
                past_key_values=think_cache,
                use_cache=True,
            )

        num_candidates = len(options)
        decision_past_len = think_cache.get_seq_length()
        conclusion_logits = conclusion_out.logits[:, -1:, :]
        expanded_cache = replicate_dynamic_cache(
            think_cache,
            num_lanes=num_candidates,
            contiguous=use_contiguous_cache,
        )

        eos = tokenizer.eos_token or ""
        candidate_continuations = [f"{option}.{eos}" for option in options]
        candidate_inputs = tokenizer(
            candidate_continuations,
            return_tensors="pt",
            padding=True,
        ).to(target_device)
        candidate_mask, candidate_positions = build_branch_attention_and_positions(
            past_len=decision_past_len,
            branch_attention_mask=candidate_inputs.attention_mask,
            device=torch.device(target_device),
        )

        with torch.no_grad():
            candidate_out = model(
                input_ids=candidate_inputs.input_ids,
                attention_mask=candidate_mask,
                position_ids=candidate_positions,
                past_key_values=expanded_cache,
                use_cache=False,
            )

        option_logits = torch.cat(
            [
                conclusion_logits.expand(num_candidates, 1, -1),
                candidate_out.logits[:, :-1, :],
            ],
            dim=1,
        )
        option_log_probs = torch.log_softmax(option_logits, dim=-1)
        option_targets = candidate_inputs.input_ids.unsqueeze(-1)
        token_log_probs = option_log_probs.gather(dim=-1, index=option_targets).squeeze(-1)
        scoring_mask = candidate_inputs.attention_mask
        token_counts = scoring_mask.sum(dim=-1).clamp(min=1)
        sequence_scores = (token_log_probs * scoring_mask).sum(dim=-1) / token_counts
        return torch.softmax(sequence_scores, dim=-1).detach().to("cpu", torch.float32)

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
                raise ValueError("SequenceThinkingArbiter requires a dedicated causal model.")
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

        for field_key, primitive in ambiguous_fields.items():
            options = primitive.options
            prompt_label = primitive.prompt or primitive.key or primitive.kind

            think_prompt = _build_thinking_prompt(
                branch_suffix=branch_suffix,
                prompt_label=prompt_label,
                options=options,
            )

            # -------------------------------------------------------------
            # STEP 1: Feed Thinking Prompt into Cloned KV-Cache
            # -------------------------------------------------------------
            think_cache = replicate_dynamic_cache(
                active_cache,
                num_lanes=1,
                contiguous=use_contiguous_cache,
            )
            think_inputs = tokenizer(think_prompt, return_tensors="pt").to(target_device)

            comb_mask, pos_ids = build_branch_attention_and_positions(
                past_len=active_past_len,
                branch_attention_mask=think_inputs.attention_mask,
                device=torch.device(target_device),
            )

            with torch.no_grad():
                think_out = model(
                    input_ids=think_inputs.input_ids,
                    attention_mask=comb_mask,
                    position_ids=pos_ids,
                    past_key_values=think_cache,
                    use_cache=True,
                )

            # -------------------------------------------------------------
            # STEP 2: Autoregressive Thinking Token Generation
            # -------------------------------------------------------------
            generated_ids: list[int] = []
            next_logits = think_out.logits[:, -1, :]
            think_end_token_ids = tokenizer.encode("</think>", add_special_tokens=False)

            for _ in range(self.max_thinking_tokens):
                next_token = _sample_next_token(
                    next_logits,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                reached_think_end = _ends_with_token_sequence(
                    generated_ids,
                    think_end_token_ids,
                )
                reached_eos = token_id == tokenizer.eos_token_id
                if reached_eos:
                    break

                cur_pos = torch.tensor(
                    [[think_cache.get_seq_length()]], dtype=torch.long, device=target_device
                )
                cur_mask = torch.ones(
                    (1, think_cache.get_seq_length() + 1), dtype=torch.long, device=target_device
                )

                with torch.no_grad():
                    step_out = model(
                        input_ids=next_token,
                        attention_mask=cur_mask,
                        position_ids=cur_pos,
                        past_key_values=think_cache,
                        use_cache=True,
                    )
                next_logits = step_out.logits[:, -1, :]
                if reached_think_end:
                    break

            trace_ids = (
                generated_ids[: -len(think_end_token_ids)]
                if _ends_with_token_sequence(generated_ids, think_end_token_ids)
                else generated_ids
            )
            thinking_trace = tokenizer.decode(trace_ids, skip_special_tokens=True).strip()

            # -------------------------------------------------------------
            # STEP 3: Finite candidate readout conditioned on thinking
            # -------------------------------------------------------------
            score_aggregate: CandidateScoreAggregate | None = None
            if self.readout == "letter":
                score_aggregate = self._score_letter_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    target_device=target_device,
                    think_cache=think_cache,
                    options=options,
                    use_contiguous_cache=use_contiguous_cache,
                )
                probs = (
                    torch.softmax(score_aggregate.scores, dim=-1)
                    .detach()
                    .to(
                        "cpu",
                        torch.float32,
                    )
                )
            else:
                probs = self._score_sequence_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    target_device=target_device,
                    think_cache=think_cache,
                    options=options,
                    use_contiguous_cache=use_contiguous_cache,
                )
            result = primitive.result_from_probabilities(
                probs,
                cascade_threshold=threshold,
                resolved_by=self.name,
                reasoning=thinking_trace,
            )
            if score_aggregate is not None:
                result.certainty_score = score_aggregate.robust_margin
                result.is_certain = bool(score_aggregate.robust_margin >= result.tau)
            certainty_score = (
                result.margin if result.certainty_score is None else result.certainty_score
            )
            if self.ambiguity_criterion == "entropy":
                result.is_certain = bool(
                    result.normalized_entropy is not None
                    and result.normalized_entropy <= (1.0 - result.tau)
                    and (score_aggregate is None or score_aggregate.robust_margin > 0.0)
                )
            elif self.ambiguity_criterion == "both":
                result.is_certain = bool(
                    certainty_score >= result.tau
                    and result.normalized_entropy is not None
                    and result.normalized_entropy <= (1.0 - result.tau)
                )
            resolved[field_key] = result

        return resolved
