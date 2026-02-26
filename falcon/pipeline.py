from __future__ import annotations

import re
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

logger = logging.getLogger(__name__)

from falcon.adapters import TruthfulQAAdapter, StrategyQAAdapter
from falcon.llm import LLM, UnifiedScorer
from falcon.models import NLIJudge, normalize_weight
from falcon.rewriter import RewriteConfig, rewrite_claims
from falcon.solver import FalconSolver


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
    """Generic text normalization for exact match."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()


def extract_yes_no(text: str) -> str:
    """Extract yes/no answer from text for StrategyQA.

    Looks for explicit yes/no tokens, handling common patterns like:
    - "Yes, because..."
    - "The answer is no."
    - "No."

    Returns: "yes", "no", or original text if unclear.
    """
    text_lower = text.strip().lower()

    # Direct single-word answers
    if text_lower in ("yes", "no"):
        return text_lower

    # Look for explicit patterns
    yes_patterns = [
        r'\byes\b',
        r'\banswer is yes\b',
        r'\baffirmative\b',
        r'\bcorrect\b',
        r'\btrue\b',
    ]

    no_patterns = [
        r'\bno\b',
        r'\banswer is no\b',
        r'\bnegative\b',
        r'\bincorrect\b',
        r'\bfalse\b',
    ]

    # Count occurrences
    yes_count = sum(1 for p in yes_patterns if re.search(p, text_lower))
    no_count = sum(1 for p in no_patterns if re.search(p, text_lower))

    # Prefer early occurrence (first sentence)
    first_sentence = text_lower.split('.')[0] if '.' in text_lower else text_lower

    if re.search(r'\byes\b', first_sentence):
        return "yes"
    if re.search(r'\bno\b', first_sentence):
        return "no"

    # Fall back to counts
    if yes_count > no_count:
        return "yes"
    if no_count > yes_count:
        return "no"

    # Uncertain - return normalized original
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
            parts = re.split(r"\s+(?:and|but|while|although|however)\s+", sent, flags=re.IGNORECASE)

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
    default_weight: float = 1.0
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
    max_pairwise: int = 496
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
    """Simple baseline: greedily keep high-weight claims while avoiding contradictions.

    This is intentionally *not* optimal. It's a fast heuristic baseline that matches the
    CS224N "greedy consistency filter" idea used in many proposals.

    We sort by descending weight (stable tie-break by index) and add a claim if it does not
    contradict any already-selected claim above the contradiction threshold `tau`.
    """
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

    # Return indices in original order for readability
    return sorted(selected)


def run_falcon_on_text(
    text: str,
    nli: NLIJudge,
    solver_cfg: Dict,
    claim_cfg: Dict,
    llm: Optional[LLM] = None,
    rewrite_cfg: Optional[Dict] = None,
) -> Tuple[str, FalconRunStats, Dict[Tuple[int, int], float], List[str], List[float]]:
    t0 = time.time()
    claims = extract_claims(
        text,
        split_on_conjunctions=bool(claim_cfg.get("split_on_conjunctions", True)),
        min_len_chars=int(claim_cfg.get("min_len_chars", 12)),
        max_claims=int(claim_cfg.get("max_claims", 32)),
    )

    if len(claims) <= 1:
        return (
            text.strip(),
            FalconRunStats(len(claims), 0, 0, 0, 0.0, time.time() - t0, False),
            {},
            claims,
            [1.0] * len(claims),
        )

    tau = float(solver_cfg.get("tau", 0.7))
    max_pairwise = int(solver_cfg.get("max_pairwise", 496))

    P = build_pairwise_P(nli, claims, max_pairwise)
    contra_before = sum(1 for v in P.values() if v > tau)

    weights = compute_claim_weights(claims, llm, str(solver_cfg.get("weight_source", "auto")))

    falcon_solver = FalconSolver(
        nli,
        tau=tau,
        mode=str(solver_cfg.get("mode", "hard")),
        lambda_penalty=float(solver_cfg.get("lambda_penalty", 1.0)),
        max_pairwise=max_pairwise,
    )

    res = falcon_solver.solve(claims, weights, P)
    selected_indices = set(res.selected_indices)
    selected_claims = [claims[i] for i in res.selected_indices]

    contra_after = sum(
        1 for (i, j), v in P.items()
        if i in selected_indices and j in selected_indices and v > tau
    )

    rewritten_text = None
    rewrite_cfg = rewrite_cfg or {}
    if rewrite_cfg.get("enabled", False) and llm:
        rcfg = RewriteConfig(
            enabled=True,
            style=rewrite_cfg.get("style", "concise"),
            preserve_facts=bool(rewrite_cfg.get("preserve_facts", True)),
            max_output_tokens=int(rewrite_cfg.get("max_output_tokens", 200)),
        )
        rewritten_text = rewrite_claims(selected_claims, llm, rcfg)

    output = rewritten_text if rewritten_text else " ".join(selected_claims)

    stats = FalconRunStats(
        n_claims=len(claims),
        n_pairs=len(P),
        contradictions_before=contra_before,
        contradictions_after=contra_after,
        solve_seconds=res.solve_seconds,
        total_seconds=time.time() - t0,
        rewritten=bool(rewritten_text),
    )
    return output, stats, P, claims, weights


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
) -> Dict:
    results = {"dataset": dataset_name, "split": split, "examples": [], "aggregate": {}}
    ds_name = (dataset_name or "").lower()

    # Always load via adapters (prevents dataset-script issues and keeps logic centralized)
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
    metrics = {
        "em_raw": 0,
        "em_greedy": 0,
        "em_falcon": 0,
        "latency": 0.0,
        "solve": 0.0,
        "cb": 0,
        "ca_greedy": 0,
        "ca_falcon": 0,
        "rw": 0,
    }

    logger.info("Running eval on %s (%d examples)", ds_name, n)

    for i in tqdm(range(n)):
        ex = ds[i]
        q = get_q(ex)
        gold = get_gold(ex)

        raw = llm.generate(q).text if llm else get_baseline(ex)

        # Run FALCON (MILP)
        falcon_out, stats, P, claims, weights = run_falcon_on_text(
            raw, nli, solver_cfg, claim_cfg, llm, rewrite_cfg
        )

        # Greedy baseline (uses the same P, claims, weights)
        greedy_indices = greedy_filter_claims(claims, weights, P, float(solver_cfg.get("tau", 0.7)))
        greedy_claims = [claims[i] for i in greedy_indices]
        greedy_out = " ".join(greedy_claims).strip() if greedy_claims else raw.strip()

        greedy_contra = 0
        if greedy_indices:
            tau = float(solver_cfg.get("tau", 0.7))
            sset = set(greedy_indices)
            greedy_contra = sum(
                1 for (i, j), v in P.items()
                if i in sset and j in sset and float(v) > tau
            )

        # Dataset-specific normalization
        if "strategy" in ds_name:
            # StrategyQA: extract yes/no tokens
            norm = extract_yes_no if em_normalize else (lambda x: x.strip())
        else:
            # TruthfulQA and others: generic normalization
            norm = normalize_text if em_normalize else (lambda x: x.strip())

        raw_ok = norm(raw) == norm(gold)
        greedy_ok = norm(greedy_out) == norm(gold)
        falcon_ok = norm(falcon_out) == norm(gold)

        metrics["em_raw"] += int(raw_ok)
        metrics["em_greedy"] += int(greedy_ok)
        metrics["em_falcon"] += int(falcon_ok)
        metrics["latency"] += stats.total_seconds
        metrics["solve"] += stats.solve_seconds
        metrics["cb"] += stats.contradictions_before
        metrics["ca_greedy"] += greedy_contra
        metrics["ca_falcon"] += stats.contradictions_after
        metrics["rw"] += int(stats.rewritten)

        results["examples"].append({
            "q": q,
            "gold": gold,
            "raw": raw,
            "greedy": greedy_out,
            "falcon": falcon_out,
            "em_greedy": int(greedy_ok),
            "em_falcon": int(falcon_ok),
            "stats": stats.__dict__,
        })

    if n > 0:
        results["aggregate"] = {
            "em_raw": metrics["em_raw"] / n,
            "em_greedy": metrics["em_greedy"] / n,
            "em_falcon": metrics["em_falcon"] / n,
            "avg_latency_s": metrics["latency"] / n,
            "avg_solve_s": metrics["solve"] / n,
            "avg_contradictions_before": metrics["cb"] / n,
            "avg_contradictions_after_greedy": metrics["ca_greedy"] / n,
            "avg_contradictions_after_falcon": metrics["ca_falcon"] / n,
            "rewrite_rate": metrics["rw"] / n,
        }

    return results
