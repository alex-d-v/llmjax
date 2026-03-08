# ── Stage 1: Build ──
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install huggingface_hub

COPY . .

# ── Stage 2: Runtime ──
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages and app code
COPY --from=builder /install /usr/local
COPY --from=builder /app /app

# Note: jax-metal only works on macOS. For cloud deploy, swap to:
#   pip install jax[cpu]   (CPU-only)
#   pip install jax[cuda12] (NVIDIA GPU)
# The Dockerfile defaults to CPU JAX for portability.
RUN pip install --no-cache-dir jax flax optax orbax-checkpoint \
    tiktoken datasets fastapi "uvicorn[standard]" pyyaml

EXPOSE 8000

# Default: serve the model. Override CMD to train instead.
# Expects checkpoint mounted at /app/checkpoints/
CMD ["python", "-m", "scripts.run_serve", "--checkpoint", "checkpoints/latest"]