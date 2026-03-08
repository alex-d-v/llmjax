"""FastAPI inference server."""

from fastapi import FastAPI
from pydantic import BaseModel
from src.generate import generate


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50


class GenerateResponse(BaseModel):
    text: str
    prompt: str
    tokens_generated: int


def create_app(model, params, tokenizer, cfg: dict) -> FastAPI:
    app = FastAPI(title="LLM-JAX", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "llm-jax"}

    @app.post("/generate", response_model=GenerateResponse)
    def gen(req: GenerateRequest):
        text = generate(
            model=model,
            params=params,
            tokenizer=tokenizer,
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            max_seq_len=cfg["max_seq_len"],
        )
        return GenerateResponse(
            text=text,
            prompt=req.prompt,
            tokens_generated=len(tokenizer.encode(text)) - len(tokenizer.encode(req.prompt)),
        )

    return app