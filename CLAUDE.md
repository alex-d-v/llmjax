# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A GPT-style language model built from scratch with JAX and Flax Linen. Decoder-only transformer (~58M params default) trained on TinyStories, with FastAPI serving.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data preparation (downloads TinyStories, tokenizes, saves to data/)
python scripts/prepare_data.py --config configs/default.yaml

# Training
python -m scripts.run_train --config configs/default.yaml

# Tests
python -m pytest tests/
python -m pytest tests/test_model.py::test_forward_pass  # single test

# Text generation (CLI)
python src/generate.py --checkpoint checkpoints/step_50000 --prompt "Once upon a time"

# API server
python scripts/run_serve.py --checkpoint checkpoints/step_50000
# POST http://localhost:8000/generate, GET /health
```

## Architecture

**Core modules** (`src/`):
- `model.py` — Transformer architecture: `GPTModel` → `TransformerBlock` → `CausalSelfAttention` + `FeedForward`. Uses RoPE (rotary embeddings), RMSNorm (pre-norm), SwiGLU FFN. All Flax `nn.Module` subclasses.
- `train.py` — Training loop with custom `TrainState` (registered as JAX pytree). AdamW + linear warmup + cosine decay via Optax. Orbax checkpointing.
- `data.py` — `prepare_dataset()` downloads/tokenizes HF datasets to `.npy` shards. `DataLoader` is an infinite random-batch generator.
- `tokenizer.py` — Thin wrapper around tiktoken (GPT-2 BPE encoding).
- `generate.py` — Autoregressive generation with temperature scaling and top-k sampling. Also runnable as CLI.

**Serving** (`serve/api.py`): FastAPI app with `/generate` and `/health` endpoints.

**Scripts** (`scripts/`): Entry points for data prep, training, and serving.

**Config** (`configs/default.yaml`): All hyperparameters (model dims, training schedule, data, serving).

## Key Patterns

- JAX functions use `@jax.jit` with `static_argnums` for non-array args (e.g., model object)
- `TrainState` is a custom pytree — `tx` (optimizer) is auxiliary/static data; `params`, `opt_state`, `step` are dynamic children
- Currently forces CPU backend (`jax_platforms=cpu`) because JAX Metal is experimental on Apple Silicon
- Data stored as uint16 numpy arrays; DataLoader samples random windows on the fly
- All config flows through a single YAML dict — no hardcoded hyperparameters
