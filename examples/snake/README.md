# 🐍 Autonomous Snake Game with Zero-Token Decisions

This example demonstrates how to turn `Qwen/Qwen2.5-1.5B-Instruct` into a real-time game agent using **Jeeva's Parallel Context Slicing (PCD)** and **BrierScoreLoss**.

---

## ⚡ Why Jeeva for Game AI?

Traditional LLM game loops ask the model to generate text character-by-character (e.g. `{"direction": "UP"}`), suffering from:
- High latency (**300ms – 1,200ms** per move) making real-time control impossible.
- Random JSON syntax errors and hallucinated keys.
- Overconfident moves even when facing lethal obstacles.

**With Jeeva Parallel Context Slicing:**
- **Zero tokens generated**: The board state is prefilled once into a shared KV cache.
- **Single-lane live readout**: The model chooses only the direction; collision safety is computed exactly by the game engine without a redundant inference lane.
- **Low-latency control**: Fast single-lane zero-token execution on Apple Silicon (`mps`).
- **Explicit scoring evidence**: Live distributions and margins, with calibrated certainty only after held-out validation.

---

## Directory structure

- `game.py`: Core Snake engine, collision mechanics, and state serializer.
- `generate_data.py`: Algorithmic BFS shortest-path oracle and single-field dataset synthesizer.
- `train_snake.py`: Optional causal fitting with the same `DirectionDecision` used by play.
- `train_snake_encoder.py`: ModernBERT fitting for the low-latency encoder path.
- `compare_models.py`: Matched held-out comparison of zero-shot causal and fitted encoder models.
- `play_snake.py`: Interactive CLI game with ANSI graphics and live telemetry.

---

## Run the game

Use the default zero-shot causal model:

```bash
PYTHONPATH=. uv run python -m examples.snake.play_snake \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

### Device control

Force execution on Apple Silicon GPU (`mps`) or CPU:

```bash
# Apple Silicon GPU
PYTHONPATH=. uv run python -m examples.snake.play_snake \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device mps

# CPU
PYTHONPATH=. uv run python -m examples.snake.play_snake \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cpu
```

### Step-by-step inspection

Advance move-by-move by pressing Enter to inspect the exact prompt, candidate probabilities, and decisions:

```bash
PYTHONPATH=. uv run python -m examples.snake.play_snake \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --step
```

### Fine-tuned artifact

```bash
PYTHONPATH=. uv run python -m examples.snake.play_snake \
    --model examples/snake/checkpoints/snake_adapter
```

### Restarting and food randomization

After a collision, press **Enter** or **R** to start a new game without reloading or
warming the model. Press **Q** to quit. Each restarted game advances the random sequence
and explicitly avoids reusing the previous session's initial food position.
By default a session is capped at 150 moves; change the bound with `--steps`.

Game setup is random by default. Pass `--seed` only when you need a reproducible sequence:

```bash
PYTHONPATH=. uv run python -m examples.snake.play_snake --seed 42
```

### Train and compare the causal and encoder paths

Both trainers and the game use the same one-field `DirectionDecision`. Collision safety
remains an exact game-engine check rather than a second model output.

```bash
PYTHONPATH=. uv run python -m examples.snake.generate_data

# Optional fitted causal artifact
PYTHONPATH=. uv run python -m examples.snake.train_snake

# Required fitted encoder artifact for the comparison
PYTHONPATH=. uv run python -m examples.snake.train_snake_encoder
```

Run both the off-the-shelf causal model and fitted encoder over the exact same held-out
JSONL cases:

```bash
PYTHONPATH=. uv run python -m examples.snake.compare_models \
    --causal-model Qwen/Qwen2.5-0.5B-Instruct \
    --encoder-artifact examples/snake/checkpoints/snake_encoder
```

The command prints the dataset SHA-256, directional accuracy score, throughput, and p50/p95
wall-clock decision latency measured on the current machine after one warm-up decision.
These are local measurements, not universal performance claims. No benchmark result is
checked into the repository.

`--delay` is a minimum frame period, not an additional post-inference sleep. If inference
already takes longer than the requested period, the next frame starts immediately. The HUD
reports measured frame rate and displays prefill, dispatch, and total latency. On MPS,
`Total` is the synchronized authoritative measurement; individual stage values are useful
diagnostics but may under-report asynchronous GPU work.
