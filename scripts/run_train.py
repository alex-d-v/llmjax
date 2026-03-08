#!/usr/bin/env python3
"""Entry point for training."""

import argparse
import yaml

# Force CPU — JAX's Metal backend is experimental and doesn't support all ops yet
import jax
jax.config.update("jax_platforms", "cpu")
from src.model import GPTModel, ModelConfig, create_model
from src.data import DataLoader
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"JAX devices: {jax.devices()}")
    print(f"Config: {cfg}")

    # Build model
    model_cfg = ModelConfig(**{
        k: cfg[k] for k in ModelConfig.__dataclass_fields__ if k in cfg
    })
    rng = jax.random.PRNGKey(cfg["seed"])
    model, variables = create_model(model_cfg, rng)

    # Data loaders
    rng, data_rng1, data_rng2 = jax.random.split(rng, 3)
    train_loader = DataLoader(
        "data/train.npy", cfg["batch_size"], cfg["max_seq_len"], data_rng1
    )
    val_loader = DataLoader(
        "data/val.npy", cfg["batch_size"], cfg["max_seq_len"], data_rng2
    )

    # Train
    train(model, variables, train_loader, val_loader, cfg)


if __name__ == "__main__":
    main()