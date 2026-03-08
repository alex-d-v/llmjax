#!/usr/bin/env python3
"""Upload trained model to HuggingFace Hub."""

import argparse
import json
import yaml
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="e.g. checkpoints/step_50000")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--repo_id", required=True, help="e.g. yourname/llm-jax-small")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Create repo on Hub
    api = HfApi()
    create_repo(args.repo_id, private=args.private, exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{args.repo_id}")

    # Prepare upload directory
    upload_dir = Path("_hub_upload")
    upload_dir.mkdir(exist_ok=True)

    # Copy checkpoint params
    ckpt_src = Path(args.checkpoint) / "params"
    ckpt_dst = upload_dir / "params"
    if ckpt_dst.exists():
        shutil.rmtree(ckpt_dst)
    shutil.copytree(ckpt_src, ckpt_dst)

    # Save config
    with open(upload_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Model card
    card = f"""---
tags:
  - jax
  - flax
  - language-model
  - gpt
  - from-scratch
license: mit
---

# llm-jax

A small GPT-style language model built from scratch with JAX/Flax.

## Model Details

| Param | Value |
|-------|-------|
| Parameters | ~{cfg['d_model']}d / {cfg['n_layers']}L / {cfg['n_heads']}H |
| Vocab Size | {cfg['vocab_size']} |
| Context Length | {cfg['max_seq_len']} |
| Tokenizer | tiktoken ({cfg.get('tokenizer_name', 'gpt2')}) |

## Usage

```bash
git clone https://huggingface.co/{args.repo_id}
cd {args.repo_id.split('/')[-1]}
python -m src.generate --checkpoint ./params --prompt "Hello world"
```

## Training

Trained on {cfg['dataset']} for {cfg['max_steps']} steps with batch size {cfg['batch_size']}.
"""
    with open(upload_dir / "README.md", "w") as f:
        f.write(card)

    # Upload everything
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.repo_id,
        commit_message=f"Upload llm-jax checkpoint",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")

    # Cleanup
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    main()