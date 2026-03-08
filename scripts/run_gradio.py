#!/usr/bin/env python3
"""Launch the Gradio web interface."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import yaml
import jax
from pathlib import Path
from src.model import ModelConfig, create_model
from src.train import load_checkpoint
from src.tokenizer import Tokenizer
from serve.gradio_app import create_gradio_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. checkpoints/step_50000")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"JAX devices: {jax.devices()}")

    model_cfg = ModelConfig(**{
        k: cfg[k] for k in ModelConfig.__dataclass_fields__ if k in cfg
    })
    rng = jax.random.PRNGKey(cfg["seed"])
    model, variables = create_model(model_cfg, rng)
    ckpt_path = str(Path(args.checkpoint).resolve() / "params")
    params = load_checkpoint(ckpt_path, variables["params"])
    tok = Tokenizer(cfg.get("tokenizer_name", "gpt2"))

    print(f"Starting Gradio on http://localhost:{args.port}")
    app = create_gradio_app(model, params, tok, cfg)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()