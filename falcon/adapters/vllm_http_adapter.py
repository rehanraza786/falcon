from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import requests

from falcon.llm import Generation, LLM


@dataclass
class VLLMHTTPConfig:
    base_url: str          # e.g., http://localhost:8000/v1
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 256
    request_logprobs: bool = False

    # Optional: many servers require top_logprobs for useful token logprobs
    top_logprobs: int = 0


class VLLMHTTPAdapter(LLM):
    """Adapter for OpenAI-compatible /v1/chat/completions endpoints (vLLM/TGI/LM Studio)."""

    def __init__(self, cfg: VLLMHTTPConfig):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self._last_token_logprobs: Optional[List[float]] = None

    def generate(self, prompt: str) -> Generation:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        if self.cfg.request_logprobs:
            payload["logprobs"] = True
            if self.cfg.top_logprobs and int(self.cfg.top_logprobs) > 0:
                payload["top_logprobs"] = int(self.cfg.top_logprobs)

        r = requests.post(self.url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        text = data["choices"][0]["message"]["content"].strip()

        token_logprobs: Optional[List[float]] = None
        try:
            lp = data["choices"][0].get("logprobs", None)
            if lp and isinstance(lp, dict):
                content = lp.get("content", None)
                if isinstance(content, list):
                    vals: List[float] = []
                    for t in content:
                        v = t.get("logprob", None)
                        if isinstance(v, (int, float)):
                            vals.append(float(v))
                    if vals:
                        token_logprobs = vals
        except Exception:
            token_logprobs = None

        self._last_token_logprobs = token_logprobs

        return Generation(
            text=text,
            meta={
                "provider": "vllm_http",
                "token_logprobs": token_logprobs,
            },
        )

    def score_text(self, text: str) -> Optional[float]:
        # Not implemented: vLLM-style endpoints typically don't provide a clean scoring endpoint.
        return None

    def score_tokens(self, text: str) -> Optional[List[float]]:
        # Best-effort: return last generation’s token logprobs if enabled.
        return self._last_token_logprobs
