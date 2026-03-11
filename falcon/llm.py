from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import math


@dataclass
class Generation:
    text: str
    meta: Dict[str, Any] | None = None


@runtime_checkable
class LLM(Protocol):
    """
    Provider-agnostic LLM interface.
    Adapters should implement at least generate().

    Optional scoring methods:
      - score_text: returns a scalar confidence-like score
      - score_tokens: returns token-level logprobs (or similar)
    """

    def generate(self, prompt: str) -> Generation:
        ...

    def score_text(self, text: str) -> Optional[float]:
        ...

    def score_tokens(self, text: str) -> Optional[List[float]]:
        ...


class NoLLM:
    """
    A stub LLM that disables generation/scoring.

    Optional improvement:
      - We raise on generate() so misconfiguration is obvious (e.g., rewrite enabled
        but no LLM configured), rather than silently producing empty outputs.
    """

    def generate(self, prompt: str) -> Generation:
        raise RuntimeError(
            "NoLLM.generate() was called, but no LLM adapter is configured. "
            "Either disable rewrite (rewrite.enabled: false) or configure llm.enabled/provider."
        )

    def score_text(self, text: str) -> Optional[float]:
        return None

    def score_tokens(self, text: str) -> Optional[List[float]]:
        return None


class UnifiedScorer:
    """
    Best-effort unified scoring for claim weighting.

    Precedence:
      1) llm.score_text(text) if provided
      2) llm.score_tokens(text) -> exp(mean(logprob))
      3) fallback None (caller decides default)
    """

    def __init__(self, llm: Optional[LLM]):
        self.llm = llm

    def score(self, text: str) -> Optional[float]:
        if self.llm is None:
            return None

        # 1) Scalar scoring if available
        try:
            s = self.llm.score_text(text)
            if s is not None and isinstance(s, (int, float)) and math.isfinite(float(s)):
                return float(s)
        except Exception:
            pass

        # 2) Token logprobs if available -> exp(mean logprob)
        try:
            toks = self.llm.score_tokens(text)
            if toks:
                vals = [float(x) for x in toks if isinstance(x, (int, float)) and math.isfinite(float(x))]
                if vals:
                    return float(math.exp(sum(vals) / len(vals)))
        except Exception:
            pass

        return None