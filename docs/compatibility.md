# Compatibility Contract

Jeeva 0.1 publishes a capability contract, not a checkpoint allowlist. The project test suite never downloads, executes, or fine-tunes pretrained checkpoints. A model family name below is an intended integration candidate, not evidence that a particular repository, revision, quantization, chat template, or hardware combination was validated by the Jeeva project.

Applications must pin immutable model and tokenizer revisions, evaluate the complete decision path on their own held-out data and target hardware, and promote the saved Jeeva artifact they actually evaluated.

## Runtime contract

| Path | Required model/tokenizer capabilities | Automated evidence | Intended candidates |
|---|---|---|---|
| Causal letter decisions | Transformers causal or image-to-text model; tokenizer chat template; bare candidate letters that each encode to exactly one token; logits for the next-token branch; compatible dynamic-cache behavior | Mocked public `DecisionModel` lifecycle, candidate-token validation, prompt rendering, permutation balancing, cache-shape, and synthetic numerical tests | Qwen 2.5, Qwen 3, Gemma, Llama |
| Causal bilinear/contextual decisions | Causal requirements plus hidden states with a stable hidden dimension; PEFT-compatible linear modules when fitting adapters | Mocked fit/save/reload lifecycle and synthetic head/loss tests | Qwen 2.5, Qwen 3, Gemma, Llama |
| Encoder candidate-marker decisions | `model_type="modernbert"`; fast tokenizer with a mask token; hidden states; compatible 4D attention masks and position IDs; declared maximum context length | Synthetic packed-row, candidate-isolation, head, fit/save/reload, and continuation tests | ModernBERT |
| Vision-language decisions | Model config explicitly declares image components; `AutoProcessor` exposes `image_processor` and tokenizer; model accepts processor-produced image tensors | Processor/error propagation and multimodal input/security tests with mocks and local images | Application-selected; experimental |

## Package and platform contract

| Surface | 0.1 contract |
|---|---|
| Python | Metadata and CI matrix target CPython 3.10 through 3.14. |
| Core dependencies | NumPy, PyTorch, Safetensors, and Transformers are direct dependencies. |
| Training | Install `jeeva-ai[training]`; PEFT is required for adapter artifacts and adapter fitting. |
| QLoRA | Install `jeeva-ai[qlora]`; requires a supported MPS or CUDA device and a working bitsandbytes/PyTorch combination. It is not part of the platform-neutral wheel smoke gate. |
| Serving | Install `jeeva-ai[serve]`; one process owns one startup model/artifact and one schema. |
| Cloud arbitration | Install `jeeva-ai[cloud]` when Jeeva constructs the OpenAI client; injected clients do not require the SDK. |
| Multimodal | Install `jeeva-ai[multimodal]`; image support is experimental and processor failures are fatal rather than downgraded to text-only. |
| Devices and dtypes | CPU, MPS, and CUDA selection plus float32, float16, and bfloat16 are exposed. Availability and correctness depend on the pinned PyTorch/model combination and must be validated by the application. |

## Application qualification

Before calling a checkpoint compatible:

1. Pin `revision` and, when different, `tokenizer_revision` to immutable commits.
2. Confirm every letter verbalizer is one token and the model's chat template renders the documented `Answer: ` boundary.
3. Exercise every schema, candidate count, context modality, dtype, and device used in production.
4. Fit and reload the exact adapter/full artifact form that will be deployed.
5. Measure held-out field metrics, accepted precision, review rate, latency, and memory on the deployment hardware.
6. Verify ambiguous results reach a human or configured arbiter rather than being silently accepted.
7. Save, evaluate, and promote the same immutable Jeeva artifact directory.

Report checkpoint-specific failures with the exact repository ID, resolved model/tokenizer commits, Jeeva artifact manifest, Python/PyTorch/Transformers versions, device, dtype, and a minimal non-sensitive reproduction.
