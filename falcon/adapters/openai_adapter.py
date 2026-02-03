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
    """OpenAI adapter using the official Python SDK (Responses API)."""

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
        kwargs = {
            "model": self.cfg.model,
            "input": prompt,
            "temperature": self.cfg.temperature,
            "max_output_tokens": self.cfg.max_tokens,
        }
        if self.cfg.request_logprobs:
            kwargs["logprobs"] = True
            if self.cfg.top_logprobs:
                kwargs["top_logprobs"] = self.cfg.top_logprobs

        resp = self.client.responses.create(**kwargs)

        # Extract text
        text = ""
        try:
            text = resp.output_text
        except Exception:
            try:
                for item in resp.output:
                    if getattr(item, "type", None) == "message":
                        for c in item.content:
                            if getattr(c, "type", None) in ("output_text", "text"):
                                text += getattr(c, "text", "")
            except Exception:
                text = ""

        text = (text or "").strip()

        # Best-effort token logprobs extraction
        token_logprobs: Optional[List[float]] = None
        try:
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        lp = getattr(c, "logprobs", None)
                        if lp and isinstance(lp, list):
                            vals: List[float] = []
                            for tok in lp:
                                val = getattr(tok, "logprob", None)
                                if isinstance(val, (int, float)):
                                    vals.append(float(val))
                            if vals:
                                token_logprobs = vals
                                break
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
