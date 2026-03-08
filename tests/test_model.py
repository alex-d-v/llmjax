import jax
import jax.numpy as jnp
from src.model import GPTModel, ModelConfig, create_model


def test_forward_pass():
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_heads=4,
        n_layers=2, d_ff=128, max_seq_len=32,
    )
    rng = jax.random.PRNGKey(0)
    model, variables = create_model(cfg, rng)

    x = jnp.ones((2, 32), dtype=jnp.int32)
    logits = model.apply(variables, x, deterministic=True)

    assert logits.shape == (2, 32, 256), f"Wrong shape: {logits.shape}"
    print("Forward pass OK:", logits.shape)


def test_param_count():
    cfg = ModelConfig(
        vocab_size=256, d_model=64, n_heads=4,
        n_layers=2, d_ff=128, max_seq_len=32,
    )
    rng = jax.random.PRNGKey(0)
    _, variables = create_model(cfg, rng)
    n = sum(p.size for p in jax.tree_util.tree_leaves(variables["params"]))
    assert n > 0
    print(f"Param count OK: {n:,}")


if __name__ == "__main__":
    test_forward_pass()
    test_param_count()
    print("All tests passed!")