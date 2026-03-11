from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from falcon.adapters import TruthfulQAAdapter, StrategyQAAdapter
from falcon.llm import LLM, UnifiedScorer
from falcon.metrics import compute_all_metrics
from falcon.models import NLIJudge, normalize_weight
from falcon.rewriter import RewriteConfig, rewrite_claims
from falcon.self_reflect import SelfReflectConfig, run_self_reflection
from falcon.solver import FalconSolver

logger = logging.getLogger(__name__)


@dataclass
class FalconRunStats:
    n_claims: int
    n_pairs: int
    contradictions_before: int
    contradictions_after: int
    solve_seconds: float
    total_seconds: float
    rewritten: bool


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()


def extract_yes_no(text: str) -> str:
    text_lower = text.strip().lower()

    if text_lower in ("yes", "no"):
        return text_lower

    yes_patterns = [
        r"\byes\b",
        r"\banswer is yes\b",
        r"\baffirmative\b",
        r"\bcorrect\b",
        r"\btrue\b",
    ]
    no_patterns = [
        r"\bno\b",
        r"\banswer is no\b",
        r"\bnegative\b",
        r"\bincorrect\b",
        r"\bfalse\b",
    ]

    yes_count = sum(1 for p in yes_patterns if re.search(p, text_lower))
    no_count = sum(1 for p in no_patterns if re.search(p, text_lower))

    first_sentence = text_lower.split(".")[0] if "." in text_lower else text_lower
    if re.search(r"\byes\b", first_sentence):
        return "yes"
    if re.search(r"\bno\b", first_sentence):
        return "no"

    if yes_count > no_count:
        return "yes"
    if no_count > yes_count:
        return "no"

    return normalize_text(text)


def extract_claims(
    text: str,
    split_on_conjunctions: bool = True,
    min_len_chars: int = 12,
    max_claims: int = 32,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    claims: List[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        parts = [sent]
        if split_on_conjunctions:
            parts = re.split(
                r"\s+(?:and|but|while|although|however)\s+",
                sent,
                flags=re.IGNORECASE,
            )

        for p in parts:
            p = p.strip()
            if len(p) >= min_len_chars:
                claims.append(p)

        if len(claims) >= max_claims:
            break

    return claims[:max_claims]


def compute_claim_weights(
    claims: List[str],
    llm: Optional[LLM],
    weight_source: str = "auto",
    default_weight: float = 1.0,
) -> List[float]:
    if not claims:
        return []

    ws = (weight_source or "auto").lower()
    if ws == "uniform" or llm is None:
        return [float(default_weight)] * len(claims)

    scorer = UnifiedScorer(llm)
    weights = []
    for c in claims:
        s = scorer.score(c)
        weights.append(normalize_weight(s, default=default_weight))
    return weights


def build_pairwise_P(
    nli: NLIJudge,
    claims: List[str],
    max_pairwise: int = 496,
) -> Dict[Tuple[int, int], float]:
    n = len(claims)
    if n <= 1:
        return {}

    pairs = []
    idx_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((claims[i], claims[j]))
            idx_pairs.append((i, j))
            if len(pairs) >= max_pairwise:
                break
        if len(pairs) >= max_pairwise:
            break

    probs = nli.contradiction_probs(pairs)

    P = {}
    for (i, j), p in zip(idx_pairs, probs):
        P[(i, j)] = float(p)
    return P


def greedy_filter_claims(
    claims: List[str],
    weights: List[float],
    P: Dict[Tuple[int, int], float],
    tau: float,
) -> List[int]:
    if not claims:
        return []
    if len(claims) != len(weights):
        raise ValueError("claims and weights must have the same length")

    order = sorted(range(len(claims)), key=lambda i: (-float(weights[i]), i))
    selected: List[int] = []

    def contradicts(a: int, b: int) -> bool:
        i, j = (a, b) if a < b else (b, a)
        return float(P.get((i, j), 0.0)) > float(tau)

    for i in order:
        ok = True
        for j in selected:
            if contradicts(i, j):
                ok = False
                break
        if ok:
            selected.append(i)

    return sorted(selected)


def count_selected_contradictions(
    selected_indices: List[int],
    P: Dict[Tuple[int, int], float],
    tau: float,
) -> int:
    sset = set(selected_indices)
    return sum(
        1 for (i, j), v in P.items()
        if i in sset and j in sset and float(v) > tau
    )


def _join_selected_claims(claims: List[str]) -> str:
    cleaned = [c.strip() for c in claims if c and c.strip()]
    return " ".join(cleaned).strip()


def run_falcon_on_text(
    text: str,
    nli: NLIJudge,
    solver_cfg: Dict,
    claim_cfg: Dict,
    llm: Optional[LLM] = None,
    rewrite_cfg: Optional[Dict] = None,
    question: Optional[str] = None,
    self_reflect_cfg: Optional[Dict] = None,
) -> Tuple[str, FalconRunStats, Dict[Tuple[int, int], float], List[str], List[float], Dict[str, Any]]:
    t0 = time.time()

    claims = extract_claims(
        text,
        split_on_conjunctions=bool(claim_cfg.get("split_on_conjunctions", True)),
        min_len_chars=int(claim_cfg.get("min_len_chars", 12)),
        max_claims=int(claim_cfg.get("max_claims", 32)),
    )

    if len(claims) <= 1:
        output = text.strip()
        extras = {
            "selected_claim_indices": list(range(len(claims))),
            "selected_claims": claims,
            "greedy_selected_indices": list(range(len(claims))),
            "greedy_selected_claims": claims,
            "self_reflect_output": None,
            "self_reflect_used": False,
        }
        return (
            output,
            FalconRunStats(len(claims), 0, 0, 0, 0.0, time.time() - t0, False),
            {},
            claims,
            [1.0] * len(claims),
            extras,
        )

    tau = float(solver_cfg.get("tau", 0.7))
    max_pairwise = int(solver_cfg.get("max_pairwise", 496))

    P = build_pairwise_P(nli, claims, max_pairwise)
    contra_before = sum(1 for v in P.values() if v > tau)

    weights = compute_claim_weights(
        claims,
        llm,
        str(solver_cfg.get("weight_source", "auto")),
    )

    falcon_solver = FalconSolver(
        nli,
        tau=tau,
        mode=str(solver_cfg.get("mode", "hard")),
        lambda_penalty=float(solver_cfg.get("lambda_penalty", 1.0)),
        max_pairwise=max_pairwise,
    )

    res = falcon_solver.solve(claims, weights, P)
    selected_indices = list(res.selected_indices)
    selected_claims = [claims[i] for i in selected_indices]
    contra_after = count_selected_contradictions(selected_indices, P, tau)

    greedy_indices = greedy_filter_claims(claims, weights, P, tau)
    greedy_claims = [claims[i] for i in greedy_indices]

    rewritten_text = None
    rewrite_cfg = rewrite_cfg or {}
    if rewrite_cfg.get("enabled", False) and llm:
        rcfg = RewriteConfig(
            enabled=True,
            style=rewrite_cfg.get("style", "concise"),
            preserve_facts=bool(rewrite_cfg.get("preserve_facts", True)),
            max_output_tokens=int(rewrite_cfg.get("max_output_tokens", 200)),
        )
        rewritten_text = rewrite_claims(
            selected_claims,
            llm,
            rcfg,
            question=question,
        )

    output = rewritten_text if rewritten_text else _join_selected_claims(selected_claims)

    self_reflect_cfg = self_reflect_cfg or {}
    self_reflect_output = None
    self_reflect_used = False
    if self_reflect_cfg.get("enabled", False) and llm and question:
        s_cfg = SelfReflectConfig(
            enabled=True,
            max_output_tokens=int(self_reflect_cfg.get("max_output_tokens", 256)),
            temperature=float(self_reflect_cfg.get("temperature", 0.2)),
            instruction=self_reflect_cfg.get(
                "instruction",
                "Revise the answer to remove contradictions while preserving correctness.",
            ),
        )
        self_reflect_output = run_self_reflection(
            question=question,
            answer=text,
            llm=llm,
            cfg=s_cfg,
        )
        self_reflect_used = bool(self_reflect_output)

    stats = FalconRunStats(
        n_claims=len(claims),
        n_pairs=len(P),
        contradictions_before=contra_before,
        contradictions_after=contra_after,
        solve_seconds=res.solve_seconds,
        total_seconds=time.time() - t0,
        rewritten=bool(rewritten_text),
    )

    extras = {
        "selected_claim_indices": selected_indices,
        "selected_claims": selected_claims,
        "greedy_selected_indices": greedy_indices,
        "greedy_selected_claims": greedy_claims,
        "self_reflect_output": self_reflect_output,
        "self_reflect_used": self_reflect_used,
    }

    return output, stats, P, claims, weights, extras


def run_eval(
    nli: NLIJudge,
    solver_cfg: Dict,
    claim_cfg: Dict,
    dataset_name: str = "truthfulqa",
    split: str = "validation",
    max_examples: int = 50,
    em_normalize: bool = True,
    llm: Optional[LLM] = None,
    rewrite_cfg: Optional[Dict] = None,
    self_reflect_cfg: Optional[Dict] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict:
    results: Dict[str, Any] = {
        "dataset": dataset_name,
        "split": split,
        "examples": [],
        "aggregate": {},
        "provenance": provenance or {},
    }
    ds_name = (dataset_name or "").lower()

    if "truthful" in ds_name:
        adapter = TruthfulQAAdapter(split=split)
    elif "strategy" in ds_name:
        adapter = StrategyQAAdapter(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info("Loading dataset via adapter: %s (%s)", adapter.__class__.__name__, split)
    ds = adapter.load()
    get_q = adapter.get_question
    get_gold = adapter.get_gold
    get_baseline = getattr(adapter, "get_baseline", lambda ex: "")

    n = min(int(max_examples), len(ds))
    metrics_sum = {
        "em_raw": 0.0,
        "em_greedy": 0.0,
        "em_falcon": 0.0,
        "em_self_reflect": 0.0,
        "token_f1_raw": 0.0,
        "token_f1_greedy": 0.0,
        "token_f1_falcon": 0.0,
        "token_f1_self_reflect": 0.0,
        "rougeL_raw": 0.0,
        "rougeL_greedy": 0.0,
        "rougeL_falcon": 0.0,
        "rougeL_self_reflect": 0.0,
        "latency": 0.0,
        "solve": 0.0,
        "cb": 0.0,
        "ca_greedy": 0.0,
        "ca_falcon": 0.0,
        "rw": 0.0,
        "sr": 0.0,
    }

    tau = float(solver_cfg.get("tau", 0.7))

    logger.info("Running eval on %s (%d examples)", ds_name, n)

    for i in tqdm(range(n)):
        ex = ds[i]
        q = get_q(ex)
        gold = get_gold(ex)

        raw = llm.generate(q).text if llm else get_baseline(ex)

        falcon_out, stats, P, claims, weights, extras = run_falcon_on_text(
            text=raw,
            nli=nli,
            solver_cfg=solver_cfg,
            claim_cfg=claim_cfg,
            llm=llm,
            rewrite_cfg=rewrite_cfg,
            question=q,
            self_reflect_cfg=self_reflect_cfg,
        )

        greedy_indices = extras["greedy_selected_indices"]
        greedy_claims = extras["greedy_selected_claims"]
        greedy_out = _join_selected_claims(greedy_claims) if greedy_claims else raw.strip()
        greedy_contra = count_selected_contradictions(greedy_indices, P, tau)

        self_reflect_out = extras.get("self_reflect_output") or raw

        raw_metrics = compute_all_metrics(raw, gold, dataset_name, em_normalize)
        greedy_metrics = compute_all_metrics(greedy_out, gold, dataset_name, em_normalize)
        falcon_metrics = compute_all_metrics(falcon_out, gold, dataset_name, em_normalize)
        self_reflect_metrics = compute_all_metrics(self_reflect_out, gold, dataset_name, em_normalize)

        metrics_sum["em_raw"] += raw_metrics["em"]
        metrics_sum["em_greedy"] += greedy_metrics["em"]
        metrics_sum["em_falcon"] += falcon_metrics["em"]
        metrics_sum["em_self_reflect"] += self_reflect_metrics["em"]

        metrics_sum["token_f1_raw"] += raw_metrics["token_f1"]
        metrics_sum["token_f1_greedy"] += greedy_metrics["token_f1"]
        metrics_sum["token_f1_falcon"] += falcon_metrics["token_f1"]
        metrics_sum["token_f1_self_reflect"] += self_reflect_metrics["token_f1"]

        metrics_sum["rougeL_raw"] += raw_metrics["rougeL"]
        metrics_sum["rougeL_greedy"] += greedy_metrics["rougeL"]
        metrics_sum["rougeL_falcon"] += falcon_metrics["rougeL"]
        metrics_sum["rougeL_self_reflect"] += self_reflect_metrics["rougeL"]

        metrics_sum["latency"] += stats.total_seconds
        metrics_sum["solve"] += stats.solve_seconds
        metrics_sum["cb"] += stats.contradictions_before
        metrics_sum["ca_greedy"] += greedy_contra
        metrics_sum["ca_falcon"] += stats.contradictions_after
        metrics_sum["rw"] += int(stats.rewritten)
        metrics_sum["sr"] += int(bool(extras.get("self_reflect_used", False)))

        results["examples"].append({
            "index": i,
            "question": q,
            "gold": gold,
            "raw": raw,
            "greedy": greedy_out,
            "falcon": falcon_out,
            "self_reflect": self_reflect_out,
            "raw_metrics": raw_metrics,
            "greedy_metrics": greedy_metrics,
            "falcon_metrics": falcon_metrics,
            "self_reflect_metrics": self_reflect_metrics,
            "claims": claims,
            "weights": weights,
            "stats": stats.__dict__,
            "selected_claim_indices": extras["selected_claim_indices"],
            "selected_claims": extras["selected_claims"],
            "greedy_selected_indices": extras["greedy_selected_indices"],
            "greedy_selected_claims": extras["greedy_selected_claims"],
        })

    if n > 0:
        results["aggregate"] = {
            "em_raw": metrics_sum["em_raw"] / n,
            "em_greedy": metrics_sum["em_greedy"] / n,
            "em_falcon": metrics_sum["em_falcon"] / n,
            "em_self_reflect": metrics_sum["em_self_reflect"] / n,
            "token_f1_raw": metrics_sum["token_f1_raw"] / n,
            "token_f1_greedy": metrics_sum["token_f1_greedy"] / n,
            "token_f1_falcon": metrics_sum["token_f1_falcon"] / n,
            "token_f1_self_reflect": metrics_sum["token_f1_self_reflect"] / n,
            "rougeL_raw": metrics_sum["rougeL_raw"] / n,
            "rougeL_greedy": metrics_sum["rougeL_greedy"] / n,
            "rougeL_falcon": metrics_sum["rougeL_falcon"] / n,
            "rougeL_self_reflect": metrics_sum["rougeL_self_reflect"] / n,
            "avg_latency_s": metrics_sum["latency"] / n,
            "avg_solve_s": metrics_sum["solve"] / n,
            "avg_contradictions_before": metrics_sum["cb"] / n,
            "avg_contradictions_after_greedy": metrics_sum["ca_greedy"] / n,
            "avg_contradictions_after_falcon": metrics_sum["ca_falcon"] / n,
            "rewrite_rate": metrics_sum["rw"] / n,
            "self_reflect_rate": metrics_sum["sr"] / n,
            "tau": tau,
            "lambda_penalty": float(solver_cfg.get("lambda_penalty", 1.0)),
            "solver_mode": str(solver_cfg.get("mode", "hard")),
        }

    return results