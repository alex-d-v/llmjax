"""Tokenizer wrapper — tiktoken (GPT-2 BPE) or SentencePiece."""

import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab
        self.eos_token = self.enc.eot_token

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)