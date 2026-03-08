"""GPT-style Transformer in Flax Linen."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout_rate: float = 0.1


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return x / rms * scale


def rotary_embedding(seq_len, dim, dtype=jnp.float32):
    """Compute rotary position embedding sin/cos tables."""
    pos = jnp.arange(seq_len, dtype=dtype)[:, None]
    i = jnp.arange(0, dim, 2, dtype=dtype)[None, :]
    theta = pos / jnp.power(10000.0, i / dim)
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)
    return cos, sin


def apply_rotary(x, cos, sin):
    """Apply rotary embedding to query/key tensors."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    # Broadcast cos/sin: (seq, d//2) -> (1, seq, 1, d//2) tiled for heads
    cos = cos[None, :, None, :]  # (1, seq, 1, half)
    sin = sin[None, :, None, :]
    cos = jnp.concatenate([cos, cos], axis=-1)  # (1, seq, 1, d)
    sin = jnp.concatenate([sin, sin], axis=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with rotary embeddings."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, C = x.shape
        head_dim = C // self.cfg.n_heads

        qkv = nn.Dense(3 * C, use_bias=False, name="qkv_proj")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.cfg.n_heads, head_dim)
        k = k.reshape(B, T, self.cfg.n_heads, head_dim)
        v = v.reshape(B, T, self.cfg.n_heads, head_dim)

        # Rotary embeddings
        cos, sin = rotary_embedding(T, head_dim, dtype=x.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # (B, heads, T, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = jnp.sqrt(head_dim).astype(x.dtype)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        attn = jnp.where(mask[None, None, :, :], attn, -1e9)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.cfg.dropout_rate)(attn, deterministic=deterministic)

        out = jnp.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = nn.Dense(C, use_bias=False, name="out_proj")(out)
        return out


class FeedForward(nn.Module):
    """SwiGLU feed-forward block."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        gate = nn.Dense(self.cfg.d_ff, use_bias=False, name="gate")(x)
        up = nn.Dense(self.cfg.d_ff, use_bias=False, name="up")(x)
        x = nn.silu(gate) * up
        x = nn.Dropout(self.cfg.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.cfg.d_model, use_bias=False, name="down")(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = x + CausalSelfAttention(self.cfg)(
            RMSNorm()(x), deterministic=deterministic
        )
        x = x + FeedForward(self.cfg)(
            RMSNorm()(x), deterministic=deterministic
        )
        return x


class GPTModel(nn.Module):
    """Full GPT-style language model."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        B, T = input_ids.shape

        tok_emb = nn.Embed(self.cfg.vocab_size, self.cfg.d_model, name="tok_emb")(
            input_ids
        )
        x = nn.Dropout(self.cfg.dropout_rate)(tok_emb, deterministic=deterministic)

        for i in range(self.cfg.n_layers):
            x = TransformerBlock(self.cfg, name=f"block_{i}")(
                x, deterministic=deterministic
            )

        x = RMSNorm()(x)
        logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head")(x)
        return logits


def create_model(cfg: ModelConfig, rng_key):
    """Initialize model and return (model, params)."""
    model = GPTModel(cfg)
    dummy = jnp.ones((1, cfg.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng_key, dummy, deterministic=True)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    print(f"Model initialized: {n_params / 1e6:.1f}M parameters")
    return model, variables