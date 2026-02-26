from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from falcon.llm import Generation, LLM


@dataclass
class OpenAIConfig:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 256
    request_logprobs: bool = False
    top_logprobs: int = 0


class OpenAIAdapter(LLM):
    """OpenAI adapter using the official Python SDK (Chat Completions API)."""

    def __init__(self, cfg: OpenAIConfig):
        self.cfg = cfg
        self._last_token_logprobs: Optional[List[float]] = None

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e

        api_key = cfg.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key missing. Set OPENAI_API_KEY or pass api_key in config.")
        self.client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    def generate(self, prompt: str) -> Generation:
        # Build messages for chat completion
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        if self.cfg.request_logprobs:
            kwargs["logprobs"] = True
            if self.cfg.top_logprobs and self.cfg.top_logprobs > 0:
                kwargs["top_logprobs"] = self.cfg.top_logprobs

        resp = self.client.chat.completions.create(**kwargs)

        # Extract text from chat completion response
        text = ""
        if resp.choices and len(resp.choices) > 0:
            choice = resp.choices[0]
            if choice.message and choice.message.content:
                text = choice.message.content.strip()

        # Extract token logprobs if available
        token_logprobs: Optional[List[float]] = None
        try:
            if resp.choices and len(resp.choices) > 0:
                choice = resp.choices[0]
                if hasattr(choice, 'logprobs') and choice.logprobs:
                    if hasattr(choice.logprobs, 'content') and choice.logprobs.content:
                        vals: List[float] = []
                        for token_data in choice.logprobs.content:
                            if hasattr(token_data, 'logprob'):
                                vals.append(float(token_data.logprob))
                        if vals:
                            token_logprobs = vals
        except Exception:
            token_logprobs = None

        self._last_token_logprobs = token_logprobs

        return Generation(
            text=text,
            meta={
                "provider": "openai",
                "token_logprobs": token_logprobs,
            },
        )

    def score_text(self, text: str) -> Optional[float]:
        # OpenAI hosted endpoints typically do not expose a clean scoring endpoint.
        return None

    def score_tokens(self, text: str) -> Optional[List[float]]:
        return self._last_token_logprobs
