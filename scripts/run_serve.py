#!/usr/bin/env python3
"""Launch the FastAPI inference server."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import yaml
import uvicorn
import jax
from pathlib import Path
from src.model import GPTModel, ModelConfig, create_model
from src.train import load_checkpoint
from src.tokenizer import Tokenizer
from serve.api import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig(**{
        k: cfg[k] for k in ModelConfig.__dataclass_fields__ if k in cfg
    })

    rng = jax.random.PRNGKey(cfg["seed"])
    model, variables = create_model(model_cfg, rng)
    ckpt_path = str(Path(args.checkpoint).resolve() / "params")
    params = load_checkpoint(ckpt_path, variables["params"])
    tok = Tokenizer(cfg.get("tokenizer_name", "gpt2"))

    app = create_app(model, params, tok, cfg)
    uvicorn.run(app, host="0.0.0.0", port=cfg.get("serve_port", 8000))


if __name__ == "__main__":
    main()