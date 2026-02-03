from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

from falcon.llm import Generation, LLM


@dataclass
class AnthropicConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 256


class AnthropicAdapter(LLM):
    """Anthropic adapter using the official SDK.

    Token logprobs are generally not exposed; unified scoring will fall back to uniform.
    """

    def __init__(self, cfg: AnthropicConfig):
        self.cfg = cfg
        try:
            import anthropic  # type: ignore
        except Exception as e:
            raise ImportError("anthropic package not installed. Run: pip install anthropic") from e

        api_key = cfg.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key missing. Set ANTHROPIC_API_KEY or pass api_key in config.")
        if cfg.base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=cfg.base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str) -> Generation:
        msg = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join([blk.text for blk in msg.content if getattr(blk, "type", None) == "text"]).strip()
        return Generation(text=text, meta={"provider": "anthropic"})

    def score_text(self, text: str) -> Optional[float]:
        return None

    def score_tokens(self, text: str) -> Optional[List[float]]:
        return None
