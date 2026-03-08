# LLM-JAX

A GPT-style language model built from scratch using **JAX** and **Flax Linen**. This project implements a complete pipeline — from data preparation and training to text generation, API serving, and a Gradio web interface — designed for learning modern transformer architecture patterns.

Trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for fast iteration.

## Architecture

```
Input Token IDs
       │
       ▼
┌─────────────┐
│  Token Embed │  (vocab_size → d_model)
│  + Dropout   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│     Transformer Block (×N)   │
│                              │
│  ┌────────────────────────┐  │
│  │  RMSNorm               │  │
│  │  Causal Self-Attention  │  │
│  │  ├─ RoPE Embeddings    │  │
│  │  ├─ Causal Mask        │  │
│  │  └─ Dropout            │  │
│  │  + Residual Connection  │  │
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │  RMSNorm               │  │
│  │  SwiGLU Feed-Forward   │  │
│  │  ├─ Gate + Up (SiLU)   │  │
│  │  ├─ Down Projection    │  │
│  │  └─ Dropout            │  │
│  │  + Residual Connection  │  │
│  └────────────────────────┘  │
└──────────────┬───────────────┘
               │
               ▼
        ┌────────────┐
        │   RMSNorm   │
        └──────┬─────┘
               │
               ▼
        ┌────────────┐
        │  LM Head    │  (d_model → vocab_size)
        └──────┬─────┘
               │
               ▼
          Logits (B, T, vocab_size)
```

### Key Design Choices

| Component | Choice | Why |
|-----------|--------|-----|
| Normalization | **RMSNorm** (pre-norm) | Faster than LayerNorm, stabilizes deep networks |
| Positional Encoding | **Rotary (RoPE)** | Relative positions, better generalization than absolute embeddings |
| Feed-Forward | **SwiGLU** | Gated activation (SiLU) improves over standard ReLU FFN |
| Optimizer | **AdamW** | Decoupled weight decay with linear warmup + cosine decay |
| Tokenizer | **tiktoken** (GPT-2 BPE) | Fast, battle-tested byte-pair encoding (50,257 vocab) |

### Model Configurations

**Default** (~77M parameters):

```yaml
vocab_size: 50257       # GPT-2 BPE vocabulary
d_model: 512            # hidden dimension
n_heads: 8              # attention heads
n_layers: 6             # transformer blocks
d_ff: 2048              # feed-forward inner dimension
max_seq_len: 512        # context window length
dropout_rate: 0.1
```

**Fast** (~30M parameters, for quick CPU training):

```yaml
vocab_size: 50257
d_model: 256
n_heads: 4
n_layers: 4
d_ff: 1024
max_seq_len: 128
dropout_rate: 0.1
```

## Project Structure

```
llm-jax/
├── src/
│   ├── model.py         # GPTModel, CausalSelfAttention, FeedForward, RMSNorm, RoPE
│   ├── train.py         # Training loop, TrainState, optimizer, checkpointing
│   ├── data.py          # Dataset download/tokenization, DataLoader
│   ├── tokenizer.py     # Tiktoken wrapper
│   └── generate.py      # Autoregressive generation with top-k sampling
├── serve/
│   ├── api.py           # FastAPI inference server (/generate, /health)
│   └── gradio_app.py    # Gradio web interface (chat + text completion)
├── scripts/
│   ├── prepare_data.py  # Data preparation entry point
│   ├── run_train.py     # Training entry point
│   ├── run_serve.py     # API server entry point
│   ├── run_gradio.py    # Gradio web UI entry point
│   └── upload_to_hub.py # Upload checkpoint to HuggingFace Hub
├── configs/
│   ├── default.yaml     # Full model config (~77M params, 50k steps)
│   └── fast.yaml        # Small model config (~30M params, 10k steps)
├── tests/
│   ├── test_model.py    # Forward pass and param count tests
│   └── test_generate.py # Generation smoke test
├── Dockerfile           # Docker image for serving
├── docker-compose.yaml  # Docker Compose config
├── data/                # Tokenized .npy shards (generated)
├── checkpoints/         # Orbax checkpoints (generated)
├── requirements.txt
└── pyproject.toml
```

## Getting Started

### Prerequisites

- Python 3.10+
- macOS (Apple Silicon) or Linux

### Installation

```bash
git clone https://github.com/alex-d-v/llmjax.git && cd llmjax
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Prepare Data

Downloads [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) from HuggingFace, tokenizes with GPT-2 BPE, and saves as numpy arrays.

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Outputs: `data/train.npy` and `data/val.npy`

### 2. Train

> **Important:** All scripts must be run as modules (`python -m`) from the project root.

```bash
# Full training (~77M params, 50k steps)
python -m scripts.run_train --config configs/default.yaml

# Fast training (~30M params, 10k steps, ~20 min on CPU)
python -m scripts.run_train --config configs/fast.yaml
```

Training logs loss every 100 steps, runs validation every 1000 steps (500 for fast config), and saves Orbax checkpoints periodically to `checkpoints/`.

### 3. Generate Text

```bash
python -m src.generate \
  --checkpoint checkpoints/step_10000 \
  --prompt "Once upon a time" \
  --config configs/fast.yaml \
  --max_tokens 256 \
  --temperature 0.8
```

### 4. Serve via API

```bash
python -m scripts.run_serve \
  --checkpoint checkpoints/step_10000 \
  --config configs/fast.yaml
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/generate` | Text generation |

**Example request:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100, "temperature": 0.8}'
```

**Response:**

```json
{
  "text": "Once upon a time there was a little girl...",
  "prompt": "Once upon a time",
  "tokens_generated": 42
}
```

### 5. Gradio Web Interface

```bash
pip install gradio
python -m scripts.run_gradio \
  --checkpoint checkpoints/step_10000 \
  --config configs/fast.yaml
```

Opens a web UI at `http://localhost:7860` with:
- **Chat tab** — conversational interface with adjustable temperature, top-k, and max tokens
- **Text Completion tab** — raw prompt completion
- **Model Info tab** — displays model configuration and parameter count

Add `--share` to create a public Gradio link.

### 6. Docker

```bash
# Build and run
docker compose up --build -d

# Check health
curl http://localhost:8000/health

# Mount your checkpoint
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints llm-jax
```

## Training Pipeline

```
HuggingFace Dataset
       │
       ▼
  Tokenize (tiktoken GPT-2 BPE, 50,257 vocab)
       │
       ▼
  Save as uint16 numpy arrays (95/5 train/val split)
       │
       ▼
  DataLoader (random window sampling → batches of (x, y) pairs)
       │
       ▼
  Training Loop
  ├─ Forward pass (with dropout)
  ├─ Cross-entropy loss (next-token prediction)
  ├─ Backward pass (jax.value_and_grad)
  ├─ AdamW update (linear warmup → cosine decay)
  ├─ Gradient clipping (global norm)
  └─ Periodic eval + Orbax checkpointing
```

### Training Hyperparameters (default)

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 (peak) |
| Warmup steps | 500 |
| Total steps | 50,000 |
| Batch size | 32 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (global norm) |
| LR schedule | Linear warmup → Cosine decay |

## Generation

Text generation uses **autoregressive sampling** with:

- **Temperature scaling** — controls randomness (lower = more deterministic)
- **Top-k filtering** — samples only from the k most likely tokens
- **EOS stopping** — generation halts when the end-of-text token is produced

## Tests

```bash
python -m pytest tests/                        # run all
python -m pytest tests/test_model.py -v        # model tests only
python -m pytest tests/test_generate.py -v     # generation tests only
```

## Platform Notes

- **Apple Silicon (M-series)**: JAX's Metal backend is experimental and doesn't support all operations. All scripts automatically force CPU mode (`JAX_PLATFORMS=cpu`) for reliability. GPU training is recommended on CUDA-capable hardware.
- **CUDA GPUs**: Remove the `os.environ.setdefault("JAX_PLATFORMS", "cpu")` lines and install `jax[cuda]` instead of `jax-metal`.
- **Docker**: The container runs in CPU mode by default. For GPU support, use `jax[cuda]` in the Dockerfile.

## Dependencies

| Package | Purpose |
|---------|---------|
| `jax` | Numerical computing, auto-differentiation, JIT compilation |
| `flax` | Neural network library (Linen API) |
| `optax` | Optimizers and learning rate schedules |
| `orbax-checkpoint` | Model checkpointing |
| `tiktoken` | GPT-2 BPE tokenizer |
| `datasets` | HuggingFace dataset loading |
| `fastapi` + `uvicorn` | API serving |
| `gradio` | Web interface (optional) |
| `pyyaml` | Configuration parsing |

## License

MIT
