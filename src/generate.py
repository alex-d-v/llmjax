"""Autoregressive text generation with temperature & top-k sampling."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(1, 3, 5))
def _generate_step(params, model, tokens, max_seq_len, rng, top_k, temperature):
    """Generate one next token."""
    # Truncate to max_seq_len if needed
    tokens_input = tokens[:, -max_seq_len:]
    logits = model.apply({"params": params}, tokens_input, deterministic=True)
    next_logits = logits[:, -1, :] / temperature

    # Top-k filtering
    top_vals, top_idx = jax.lax.top_k(next_logits, top_k)
    mask = jnp.full_like(next_logits, -1e9)
    mask = mask.at[:, :].set(-1e9)
    # Scatter top-k values back
    batch_idx = jnp.arange(next_logits.shape[0])[:, None]
    mask = mask.at[batch_idx, top_idx].set(top_vals)

    rng, subkey = jax.random.split(rng)
    next_token = jax.random.categorical(subkey, mask, axis=-1)[:, None]
    return next_token, rng


def generate(
    model,
    params,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    max_seq_len: int = 512,
    seed: int = 0,
):
    """Generate text from a prompt."""
    input_ids = tokenizer.encode(prompt)
    tokens = jnp.array([input_ids], dtype=jnp.int32)
    rng = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        next_token, rng = _generate_step(
            params, model, tokens, max_seq_len, rng, top_k, temperature
        )
        tokens = jnp.concatenate([tokens, next_token], axis=1)

        # Stop at EOS
        if int(next_token[0, 0]) == tokenizer.eos_token:
            break

    output_ids = tokens[0].tolist()
    return tokenizer.decode(output_ids)


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from src.model import GPTModel, ModelConfig, create_model
    from src.train import load_checkpoint
    from src.tokenizer import Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig(**{
        k: cfg[k] for k in ModelConfig.__dataclass_fields__ if k in cfg
    })
    tok = Tokenizer(cfg.get("tokenizer_name", "gpt2"))

    rng = jax.random.PRNGKey(cfg["seed"])
    model, variables = create_model(model_cfg, rng)

    ckpt_path = str(Path(args.checkpoint).resolve() / "params")
    params = load_checkpoint(ckpt_path, variables["params"])

    text = generate(
        model, params, tok, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=cfg.get("top_k", 50),
        max_seq_len=cfg["max_seq_len"],
    )
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")