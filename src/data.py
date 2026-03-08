"""Dataset loading and batching utilities."""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from datasets import load_dataset
from src.tokenizer import Tokenizer


def prepare_dataset(cfg: dict, output_dir: str = "data"):
    """Download dataset, tokenize, and save as numpy memmap shards."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    tok = Tokenizer(cfg.get("tokenizer_name", "gpt2"))

    print(f"Loading dataset: {cfg['dataset']} ...")
    ds = load_dataset(cfg["dataset"], split="train", trust_remote_code=True)

    # Tokenize all texts
    all_ids = []
    for i, example in enumerate(ds):
        text = example.get("text", "")
        if text.strip():
            ids = tok.encode(text) + [tok.eos_token]
            all_ids.extend(ids)
        if (i + 1) % 10000 == 0:
            print(f"  Tokenized {i + 1} examples ({len(all_ids)} tokens)")

    all_ids = np.array(all_ids, dtype=np.uint16)
    print(f"Total tokens: {len(all_ids):,}")

    # Split 95/5 train/val
    n = len(all_ids)
    split = int(n * 0.95)
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    train_path = out / "train.npy"
    val_path = out / "val.npy"
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    print(f"Saved {train_path} ({len(train_ids):,} tokens)")
    print(f"Saved {val_path} ({len(val_ids):,} tokens)")
    return train_path, val_path


class DataLoader:
    """Yields random batches of (input, target) token sequences."""

    def __init__(self, data_path: str, batch_size: int, seq_len: int, rng):
        self.data = np.load(data_path).astype(np.int32)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = rng

    def __iter__(self):
        return self

    def __next__(self):
        self.rng, subkey = tuple(
            __import__("jax").random.split(self.rng)
        )
        max_start = len(self.data) - self.seq_len - 1
        starts = __import__("jax").random.randint(
            subkey, (self.batch_size,), 0, max_start
        )
        starts = np.array(starts)

        x = np.stack([self.data[s : s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return jnp.array(x), jnp.array(y)