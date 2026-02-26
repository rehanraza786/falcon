"""
Ablation Study for FALCON: tau and max_claims

This script runs systematic ablations over:
1. Contradiction threshold (tau)
2. Maximum claims (n)
3. Solver mode (hard vs soft)

Outputs:
- Individual result files
- Scaling analysis (solve time vs. number of claims)
- Solver robustness metrics
"""

import json
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

from falcon.models import NLIJudge
from falcon.pipeline import run_falcon_on_text
from main import load_llm_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# Ablation parameters
TAU_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
MAX_CLAIMS_VALUES = [5, 10, 15, 20, 25]
MODES = ["hard", "soft"]

# Test dataset
TEST_TEXTS = [
    "The sky is blue. The sky is red. Water is wet.",
    "Paris is the capital of France. Paris is the capital of Germany. The Eiffel Tower is in Paris.",
    "Dogs are mammals. Dogs are reptiles. Cats are felines. Birds can fly.",
    "The Earth is round. The Earth is flat. Gravity exists. Objects fall downward.",
    "Python is a programming language. Python is a snake. Code requires syntax.",
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_ablation_tau(
    texts: List[str],
    nli: NLIJudge,
    base_cfg: dict,
    output_dir: Path
) -> Dict[str, Any]:
    """Ablate over tau threshold."""
    logger.info("Running tau ablation...")
    results = []

    for tau in TAU_VALUES:
        logger.info(f"Testing tau={tau}")
        tau_metrics = {
            "tau": tau,
            "contradictions_removed": [],
            "solve_times": [],
            "claims_kept_ratio": [],
        }

        solver_cfg = base_cfg.get("solver", {}).copy()
        solver_cfg["tau"] = tau

        for text in texts:
            output, stats, P, claims, weights = run_falcon_on_text(
                text=text,
                nli=nli,
                solver_cfg=solver_cfg,
                claim_cfg=base_cfg.get("claims", {}),
                llm=None,
                rewrite_cfg=base_cfg.get("rewrite", {}),
            )

            tau_metrics["contradictions_removed"].append(
                stats.contradictions_before - stats.contradictions_after
            )
            tau_metrics["solve_times"].append(stats.solve_seconds)
            tau_metrics["claims_kept_ratio"].append(
                stats.n_claims / len(claims) if len(claims) > 0 else 1.0
            )

        # Aggregate
        tau_metrics["avg_contradictions_removed"] = np.mean(tau_metrics["contradictions_removed"])
        tau_metrics["avg_solve_time"] = np.mean(tau_metrics["solve_times"])
        tau_metrics["avg_claims_kept"] = np.mean(tau_metrics["claims_kept_ratio"])

        results.append(tau_metrics)

    # Save results
    output_file = output_dir / "ablation_tau.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Tau ablation saved to {output_file}")
    return {"tau_ablation": results}


def run_ablation_max_claims(
    texts: List[str],
    nli: NLIJudge,
    base_cfg: dict,
    output_dir: Path
) -> Dict[str, Any]:
    """Ablate over maximum number of claims."""
    logger.info("Running max_claims ablation...")
    results = []

    for max_claims in MAX_CLAIMS_VALUES:
        logger.info(f"Testing max_claims={max_claims}")
        claim_metrics = {
            "max_claims": max_claims,
            "actual_claims": [],
            "solve_times": [],
            "num_pairs": [],
            "contradictions": [],
        }

        claim_cfg = base_cfg.get("claims", {}).copy()
        claim_cfg["max_claims"] = max_claims

        for text in texts:
            output, stats, P, claims, weights = run_falcon_on_text(
                text=text,
                nli=nli,
                solver_cfg=base_cfg.get("solver", {}),
                claim_cfg=claim_cfg,
                llm=None,
                rewrite_cfg=base_cfg.get("rewrite", {}),
            )

            claim_metrics["actual_claims"].append(stats.n_claims)
            claim_metrics["solve_times"].append(stats.solve_seconds)
            claim_metrics["num_pairs"].append(stats.n_pairs)
            claim_metrics["contradictions"].append(stats.contradictions_before)

        # Aggregate
        claim_metrics["avg_claims"] = np.mean(claim_metrics["actual_claims"])
        claim_metrics["avg_solve_time"] = np.mean(claim_metrics["solve_times"])
        claim_metrics["avg_pairs"] = np.mean(claim_metrics["num_pairs"])
        claim_metrics["avg_contradictions"] = np.mean(claim_metrics["contradictions"])

        # Scaling: solve time vs claims (linear regression)
        if len(claim_metrics["actual_claims"]) > 1:
            slope = np.polyfit(
                claim_metrics["actual_claims"],
                claim_metrics["solve_times"],
                1
            )[0]
            claim_metrics["scaling_slope"] = float(slope)

        results.append(claim_metrics)

    # Save results
    output_file = output_dir / "ablation_max_claims.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Max claims ablation saved to {output_file}")
    return {"max_claims_ablation": results}


def run_solver_robustness(
    texts: List[str],
    nli: NLIJudge,
    base_cfg: dict,
    output_dir: Path
) -> Dict[str, Any]:
    """Test solver robustness: hard vs soft mode comparison."""
    logger.info("Running solver robustness analysis...")
    results = {"hard": [], "soft": []}

    for mode in MODES:
        logger.info(f"Testing mode={mode}")

        solver_cfg = base_cfg.get("solver", {}).copy()
        solver_cfg["mode"] = mode

        mode_metrics = {
            "mode": mode,
            "solve_times": [],
            "contradictions_after": [],
            "claims_selected": [],
            "objective_values": [],
        }

        for text in texts:
            output, stats, P, claims, weights = run_falcon_on_text(
                text=text,
                nli=nli,
                solver_cfg=solver_cfg,
                claim_cfg=base_cfg.get("claims", {}),
                llm=None,
                rewrite_cfg=base_cfg.get("rewrite", {}),
            )

            mode_metrics["solve_times"].append(stats.solve_seconds)
            mode_metrics["contradictions_after"].append(stats.contradictions_after)
            mode_metrics["claims_selected"].append(stats.n_claims)

        # Aggregate
        mode_metrics["avg_solve_time"] = np.mean(mode_metrics["solve_times"])
        mode_metrics["std_solve_time"] = np.std(mode_metrics["solve_times"])
        mode_metrics["avg_contradictions_after"] = np.mean(mode_metrics["contradictions_after"])
        mode_metrics["avg_claims_selected"] = np.mean(mode_metrics["claims_selected"])

        results[mode].append(mode_metrics)

    # Save results
    output_file = output_dir / "ablation_robustness.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Robustness analysis saved to {output_file}")
    return {"robustness": results}


def generate_scaling_report(output_dir: Path):
    """Generate a summary report of scaling behavior."""
    logger.info("Generating scaling report...")

    # Load max_claims ablation
    claims_file = output_dir / "ablation_max_claims.json"
    if not claims_file.exists():
        logger.warning("Max claims ablation not found")
        return

    with open(claims_file) as f:
        claims_data = json.load(f)

    report = []
    report.append("=" * 70)
    report.append("FALCON Scaling Analysis: Solve Time vs. Number of Claims")
    report.append("=" * 70)
    report.append("")
    report.append(f"{'max_claims':<12} {'avg_claims':<12} {'avg_pairs':<12} {'avg_solve_time':<15}")
    report.append("-" * 70)

    for entry in claims_data:
        report.append(
            f"{entry['max_claims']:<12} "
            f"{entry['avg_claims']:<12.2f} "
            f"{entry['avg_pairs']:<12.2f} "
            f"{entry['avg_solve_time']:<15.4f}"
        )

    report.append("")
    report.append("Scaling Observations:")
    report.append(f"- Claims: {claims_data[0]['avg_claims']:.1f} → {claims_data[-1]['avg_claims']:.1f}")
    report.append(f"- Pairs: {claims_data[0]['avg_pairs']:.1f} → {claims_data[-1]['avg_pairs']:.1f}")
    report.append(f"- Solve time: {claims_data[0]['avg_solve_time']:.4f}s → {claims_data[-1]['avg_solve_time']:.4f}s")

    if 'scaling_slope' in claims_data[-1]:
        report.append(f"- Linear scaling slope: {claims_data[-1]['scaling_slope']:.6f} s/claim")

    report.append("=" * 70)

    report_text = "\n".join(report)
    report_file = output_dir / "scaling_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(report_text)
    logger.info(f"Scaling report saved to {report_file}")


def main():
    """Run all ablation studies."""
    logger.info("Starting FALCON ablation study...")

    # Setup
    cfg = load_config()
    output_dir = Path("outputs/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NLI
    nli_cfg = cfg.get("nli", {})
    nli = NLIJudge(
        model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-base"),
        device=nli_cfg.get("device", "auto"),
        batch_size=int(nli_cfg.get("batch_size", 8)),
    )

    all_results = {}

    # Run ablations
    all_results.update(run_ablation_tau(TEST_TEXTS, nli, cfg, output_dir))
    all_results.update(run_ablation_max_claims(TEST_TEXTS, nli, cfg, output_dir))
    all_results.update(run_solver_robustness(TEST_TEXTS, nli, cfg, output_dir))

    # Generate reports
    generate_scaling_report(output_dir)

    # Save combined summary
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"✅ Ablation study complete! Results in {output_dir}")


if __name__ == "__main__":
    main()

