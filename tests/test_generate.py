"""Smoke test for generation."""

import jax
import jax.numpy as jnp
from src.model import GPTModel, ModelConfig, create_model
from src.tokenizer import Tokenizer
from src.generate import generate


def test_generate():
    cfg = ModelConfig(
        vocab_size=50257, d_model=64, n_heads=4,
        n_layers=2, d_ff=128, max_seq_len=32,
    )
    rng = jax.random.PRNGKey(42)
    model, variables = create_model(cfg, rng)
    tok = Tokenizer("gpt2")

    text = generate(
        model, variables["params"], tok,
        prompt="Hello",
        max_new_tokens=10,
        temperature=1.0,
        top_k=10,
        max_seq_len=32,
        seed=42,
    )
    assert len(text) > len("Hello")
    print(f"Generated: {text!r}")
    print("Generation test OK!")


if __name__ == "__main__":
    test_generate()