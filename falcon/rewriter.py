from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from falcon.llm import LLM


@dataclass
class RewriteConfig:
    enabled: bool = False
    style: str = "concise"   # concise | neutral | detailed
    preserve_facts: bool = True
    max_output_tokens: int = 200


def _rewrite_prompt(
    claims: List[str],
    style: str,
    preserve_facts: bool,
    question: Optional[str] = None,
) -> str:
    style = (style or "concise").lower()
    if style not in {"concise", "neutral", "detailed"}:
        style = "concise"

    style_hint = {
        "concise": "Write one short answer paragraph.",
        "neutral": "Write one clear and neutral answer paragraph.",
        "detailed": "Write one to two coherent paragraphs with smooth transitions.",
    }[style]

    guardrail = ""
    if preserve_facts:
        guardrail = (
            "IMPORTANT: Do not add new facts, do not speculate, and do not introduce unsupported claims. "
            "Only use the information in the provided claims."
        )

    bullets = "\n".join(f"- {c.strip()}" for c in claims if c.strip())

    if question:
        return (
            "You are given an original question and a set of selected factual claims.\n"
            "Your task is to answer the question using only the selected claims.\n"
            f"{style_hint}\n"
            f"{guardrail}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Selected claims:\n"
            f"{bullets}\n\n"
            "Write a coherent final answer:"
        )

    return (
        "You are rewriting a set of selected factual claims into a coherent response.\n"
        f"{style_hint}\n"
        f"{guardrail}\n\n"
        "Claims to rewrite:\n"
        f"{bullets}\n\n"
        "Write a coherent final answer:"
    )


def rewrite_claims(
    claims: List[str],
    llm: Optional[LLM],
    cfg: RewriteConfig,
    question: Optional[str] = None,
) -> Optional[str]:
    if not cfg.enabled or llm is None:
        return None

    cleaned = [c.strip() for c in claims if c and c.strip()]
    if not cleaned:
        return None

    prompt = _rewrite_prompt(
        cleaned,
        cfg.style,
        cfg.preserve_facts,
        question=question,
    )
    gen = llm.generate(prompt)
    text = (gen.text or "").strip()
    return text if text else None