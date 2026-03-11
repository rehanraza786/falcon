"""
falcon_engine.py

Optional helper entrypoint that uses the SAME pipeline functions as main.py.
Useful if you want to import FALCON as a library-like component.
"""

from __future__ import annotations

from typing import Optional

import yaml

from falcon.models import load_nli_judge
from falcon.pipeline import run_falcon_on_text, run_eval
from falcon.llm import LLM
from main import load_llm_from_config  # reuse the factory for consistency


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse to a dictionary.")
    return cfg


def run_single(cfg_path: str, text: str) -> dict:
    cfg = load_config(cfg_path)
    nli_cfg = cfg.get("nli", {}) or {}
    nli = load_nli_judge(nli_cfg["model_name"], device=nli_cfg.get("device", "auto"))
    llm: Optional[LLM] = load_llm_from_config(cfg)

    filtered, stats, P, claims, weights = run_falcon_on_text(
        text=text,
        nli=nli,
        solver_cfg=cfg.get("solver", {}),
        claim_cfg=cfg.get("claims", {}),
        llm=llm,
        rewrite_cfg=cfg.get("rewrite", {}),
    )

    return {
        "input": text,
        "claims": claims,
        "weights": weights,
        "pairs": len(P),
        "stats": stats.__dict__,
        "output": filtered,
    }


def run_benchmark(cfg_path: str) -> dict:
    cfg = load_config(cfg_path)
    nli_cfg = cfg.get("nli", {}) or {}
    nli = load_nli_judge(nli_cfg["model_name"], device=nli_cfg.get("device", "auto"))
    llm: Optional[LLM] = load_llm_from_config(cfg)

    ev = cfg.get("eval", {}) or {}
    return run_eval(
        nli=nli,
        solver_cfg=cfg.get("solver", {}),
        claim_cfg=cfg.get("claims", {}),
        dataset_name=ev.get("dataset", "truthfulqa"),
        split=ev.get("split", "validation"),
        max_examples=int(ev.get("max_examples", 50)),
        em_normalize=bool(ev.get("em_normalize", True)),
        llm=llm,
        rewrite_cfg=cfg.get("rewrite", {}),
    )