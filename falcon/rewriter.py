from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from falcon.llm import LLM


@dataclass
class RewriteConfig:
    enabled: bool = False
    style: str = "concise"         # concise | neutral | detailed
    preserve_facts: bool = True
    max_output_tokens: int = 200


def _rewrite_prompt(claims: List[str], style: str, preserve_facts: bool) -> str:
    style = (style or "concise").lower()
    if style not in {"concise", "neutral", "detailed"}:
        style = "concise"

    style_hint = {
        "concise": "Write 1 short paragraph. Keep it tight and direct.",
        "neutral": "Write a clear paragraph in a neutral tone.",
        "detailed": "Write 1-2 paragraphs with smooth transitions and mild elaboration.",
    }[style]

    guardrail = ""
    if preserve_facts:
        guardrail = (
            "IMPORTANT: Do NOT add any new facts. Do NOT speculate. "
            "Only restate and connect the claims. If something is unclear, keep it generic."
        )

    bullets = "\n".join([f"- {c.strip()}" for c in claims if c.strip()])
    return (
        "You are rewriting a set of selected factual claims into a coherent response.\n"
        f"{style_hint}\n"
        f"{guardrail}\n\n"
        "Claims to rewrite (do not add or remove information):\n"
        f"{bullets}\n\n"
        "Output:"
    )


def rewrite_claims(
    claims: List[str],
    llm: Optional[LLM],
    cfg: RewriteConfig,
) -> Optional[str]:
    if not cfg.enabled:
        return None
    if llm is None:
        return None

    cleaned = [c.strip() for c in claims if c and c.strip()]
    if not cleaned:
        return None

    prompt = _rewrite_prompt(cleaned, cfg.style, cfg.preserve_facts)
    gen = llm.generate(prompt)
    text = (gen.text or "").strip()
    return text if text else None
