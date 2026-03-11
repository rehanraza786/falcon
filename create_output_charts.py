#!/usr/bin/env python3
"""
create_output_charts.py

Generate charts from FALCON experiment output JSON files.

Supports:
- main result charts from strategyqa_main.json and truthfulqa_main.json
- variant comparison charts from experiment_results.json
- ablation charts from ablation_results.json

Usage examples:
    python create_output_charts.py \
      --strategy outputs/strategyqa_main_openai.json \
      --truthful outputs/truthfulqa_main_openai.json \
      --experiments outputs/experiment_results_openai.json \
      --ablations outputs/ablation_results_openai.json \
      --outdir outputs/charts

    python create_output_charts.py \
      --strategy outputs/strategyqa_main.json \
      --truthful outputs/truthfulqa_main.json \
      --outdir outputs/charts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_outdir(path: str | Path) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    outpath: Path,
    rotation: int = 0,
    annotate: bool = True,
) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)

    if annotate:
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_grouped_bar_chart(
    groups: List[str],
    series: List[Tuple[str, List[float]]],
    title: str,
    ylabel: str,
    outpath: Path,
    annotate: bool = False,
) -> None:
    plt.figure(figsize=(10, 5))

    n_groups = len(groups)
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x_positions = list(range(n_groups))

    for i, (series_name, values) in enumerate(series):
        offsets = [x + (i - (n_series - 1) / 2) * width for x in x_positions]
        bars = plt.bar(offsets, values, width=width, label=series_name)

        if annotate:
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(x_positions, groups)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_line_chart(
    x_values: List[float],
    y_values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path,
    annotate: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if annotate:
        for x, y in zip(x_values, y_values):
            plt.text(x, y, f"{y:.3f}", fontsize=8, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def chart_main_results(
    strategy_data: Optional[Dict[str, Any]],
    truthful_data: Optional[Dict[str, Any]],
    outdir: Path,
) -> None:
    datasets: List[Tuple[str, Dict[str, Any]]] = []
    if strategy_data is not None:
        datasets.append(("StrategyQA", strategy_data))
    if truthful_data is not None:
        datasets.append(("TruthfulQA", truthful_data))

    if not datasets:
        return

    # Chart 1: EM comparison for each dataset
    for dataset_name, data in datasets:
        agg = data.get("aggregate", {})
        labels = ["Raw", "Greedy", "FALCON", "Self-Reflect"]
        values = [
            float(agg.get("em_raw", 0.0)),
            float(agg.get("em_greedy", 0.0)),
            float(agg.get("em_falcon", 0.0)),
            float(agg.get("em_self_reflect", 0.0)),
        ]
        save_bar_chart(
            labels=labels,
            values=values,
            title=f"{dataset_name}: Exact Match by Method",
            ylabel="Exact Match",
            outpath=outdir / f"{dataset_name.lower()}_em_bar.png",
        )

    # Chart 2: token-F1 comparison if present
    for dataset_name, data in datasets:
        agg = data.get("aggregate", {})
        if "token_f1_raw" not in agg:
            continue
        labels = ["Raw", "Greedy", "FALCON", "Self-Reflect"]
        values = [
            float(agg.get("token_f1_raw", 0.0)),
            float(agg.get("token_f1_greedy", 0.0)),
            float(agg.get("token_f1_falcon", 0.0)),
            float(agg.get("token_f1_self_reflect", 0.0)),
        ]
        save_bar_chart(
            labels=labels,
            values=values,
            title=f"{dataset_name}: Token-F1 by Method",
            ylabel="Token-F1",
            outpath=outdir / f"{dataset_name.lower()}_tokenf1_bar.png",
        )

    # Chart 3: ROUGE-L comparison if present
    for dataset_name, data in datasets:
        agg = data.get("aggregate", {})
        if "rougeL_raw" not in agg:
            continue
        labels = ["Raw", "Greedy", "FALCON", "Self-Reflect"]
        values = [
            float(agg.get("rougeL_raw", 0.0)),
            float(agg.get("rougeL_greedy", 0.0)),
            float(agg.get("rougeL_falcon", 0.0)),
            float(agg.get("rougeL_self_reflect", 0.0)),
        ]
        save_bar_chart(
            labels=labels,
            values=values,
            title=f"{dataset_name}: ROUGE-L by Method",
            ylabel="ROUGE-L",
            outpath=outdir / f"{dataset_name.lower()}_rougel_bar.png",
        )

    # Chart 4: contradiction before/after for both datasets
    groups = []
    before_vals = []
    greedy_vals = []
    falcon_vals = []

    for dataset_name, data in datasets:
        agg = data.get("aggregate", {})
        groups.append(dataset_name)
        before_vals.append(float(agg.get("avg_contradictions_before", 0.0)))
        greedy_vals.append(float(agg.get("avg_contradictions_after_greedy", 0.0)))
        falcon_vals.append(float(agg.get("avg_contradictions_after_falcon", 0.0)))

    save_grouped_bar_chart(
        groups=groups,
        series=[
            ("Before", before_vals),
            ("After Greedy", greedy_vals),
            ("After FALCON", falcon_vals),
        ],
        title="Average Contradictions Before and After Filtering",
        ylabel="Average Contradictions",
        outpath=outdir / "main_contradictions_grouped.png",
        annotate=True,
    )

    # Chart 5: rewrite and self-reflect rates
    rewrite_rates = []
    self_reflect_rates = []
    for _, data in datasets:
        agg = data.get("aggregate", {})
        rewrite_rates.append(float(agg.get("rewrite_rate", 0.0)))
        self_reflect_rates.append(float(agg.get("self_reflect_rate", 0.0)))

    save_grouped_bar_chart(
        groups=[name for name, _ in datasets],
        series=[
            ("Rewrite Rate", rewrite_rates),
            ("Self-Reflect Rate", self_reflect_rates),
        ],
        title="Rewrite and Self-Reflection Activation Rates",
        ylabel="Rate",
        outpath=outdir / "main_activation_rates_grouped.png",
        annotate=True,
    )


def chart_variant_results(experiment_data: Dict[str, Any], outdir: Path) -> None:
    summary = experiment_data.get("summary", {})
    if not summary:
        return

    variant_names = list(summary.keys())

    em_falcon = [float(summary[name].get("em_falcon", 0.0)) for name in variant_names]
    em_raw = [float(summary[name].get("em_raw", 0.0)) for name in variant_names]
    em_self = [float(summary[name].get("em_self_reflect", 0.0)) for name in variant_names]

    save_grouped_bar_chart(
        groups=variant_names,
        series=[
            ("Raw EM", em_raw),
            ("FALCON EM", em_falcon),
            ("Self-Reflect EM", em_self),
        ],
        title="Variant Comparison: Exact Match",
        ylabel="Exact Match",
        outpath=outdir / "variants_em_grouped.png",
        annotate=False,
    )

    contradictions_before = [
        float(summary[name].get("avg_contradictions_before", 0.0)) for name in variant_names
    ]
    contradictions_after = [
        float(summary[name].get("avg_contradictions_after_falcon", 0.0)) for name in variant_names
    ]

    save_grouped_bar_chart(
        groups=variant_names,
        series=[
            ("Before", contradictions_before),
            ("After FALCON", contradictions_after),
        ],
        title="Variant Comparison: Contradictions",
        ylabel="Average Contradictions",
        outpath=outdir / "variants_contradictions_grouped.png",
        annotate=False,
    )

    rewrite_rates = [float(summary[name].get("rewrite_rate", 0.0)) for name in variant_names]
    self_reflect_rates = [float(summary[name].get("self_reflect_rate", 0.0)) for name in variant_names]

    save_grouped_bar_chart(
        groups=variant_names,
        series=[
            ("Rewrite Rate", rewrite_rates),
            ("Self-Reflect Rate", self_reflect_rates),
        ],
        title="Variant Comparison: Activation Rates",
        ylabel="Rate",
        outpath=outdir / "variants_activation_rates_grouped.png",
        annotate=False,
    )


def chart_ablation_results(ablation_data: Dict[str, Any], outdir: Path) -> None:
    runs = ablation_data.get("runs", [])
    if not runs:
        return

    # Separate by mode
    for mode in ["hard", "soft"]:
        mode_runs = [r for r in runs if r.get("mode") == mode]
        if not mode_runs:
            continue

        # Sort by tau then lambda
        mode_runs = sorted(mode_runs, key=lambda r: (float(r["tau"]), float(r["lambda_penalty"])))

        labels = [f"τ={r['tau']}, λ={r['lambda_penalty']}" for r in mode_runs]
        em_vals = [float(r["summary"].get("em_falcon", 0.0)) for r in mode_runs]

        save_bar_chart(
            labels=labels,
            values=em_vals,
            title=f"Ablation Results ({mode.title()} Mode): FALCON Exact Match",
            ylabel="Exact Match",
            outpath=outdir / f"ablation_{mode}_em_bar.png",
            rotation=45,
            annotate=False,
        )

    # Line charts by tau for each lambda within each mode
    for mode in ["hard", "soft"]:
        mode_runs = [r for r in runs if r.get("mode") == mode]
        if not mode_runs:
            continue

        lambdas = sorted({float(r["lambda_penalty"]) for r in mode_runs})
        taus = sorted({float(r["tau"]) for r in mode_runs})

        plt.figure(figsize=(9, 5))
        for lam in lambdas:
            lam_runs = [r for r in mode_runs if float(r["lambda_penalty"]) == lam]
            lam_runs = sorted(lam_runs, key=lambda r: float(r["tau"]))
            x_vals = [float(r["tau"]) for r in lam_runs]
            y_vals = [float(r["summary"].get("em_falcon", 0.0)) for r in lam_runs]
            plt.plot(x_vals, y_vals, marker="o", label=f"λ={lam}")

        plt.title(f"Ablation Sweep ({mode.title()} Mode): Exact Match vs Tau")
        plt.xlabel("Tau")
        plt.ylabel("FALCON Exact Match")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"ablation_{mode}_tau_line.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Best configuration summary charts
    sorted_runs = sorted(
        runs,
        key=lambda r: float(r["summary"].get("em_falcon", 0.0)),
        reverse=True,
    )
    top_runs = sorted_runs[:8]

    labels = [r["name"] for r in top_runs]
    values = [float(r["summary"].get("em_falcon", 0.0)) for r in top_runs]

    save_bar_chart(
        labels=labels,
        values=values,
        title="Top Ablation Configurations by FALCON Exact Match",
        ylabel="Exact Match",
        outpath=outdir / "ablation_top_configs_bar.png",
        rotation=45,
        annotate=False,
    )


def chart_claim_histogram_from_examples(
    dataset_name: str,
    data: Dict[str, Any],
    outdir: Path,
) -> None:
    examples = data.get("examples", [])
    if not examples:
        return

    claim_counts = [len(ex.get("claims", [])) for ex in examples]
    max_count = max(claim_counts) if claim_counts else 0
    bins = list(range(max_count + 2))

    plt.figure(figsize=(8, 5))
    plt.hist(claim_counts, bins=bins, align="left", rwidth=0.8)
    plt.title(f"{dataset_name}: Distribution of Extracted Claim Counts")
    plt.xlabel("Number of Extracted Claims")
    plt.ylabel("Number of Examples")
    plt.xticks(range(max_count + 1))
    plt.tight_layout()
    plt.savefig(outdir / f"{dataset_name.lower()}_claim_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()


def write_summary_text(
    strategy_data: Optional[Dict[str, Any]],
    truthful_data: Optional[Dict[str, Any]],
    experiment_data: Optional[Dict[str, Any]],
    ablation_data: Optional[Dict[str, Any]],
    outdir: Path,
) -> None:
    lines: List[str] = []

    if strategy_data is not None:
        agg = strategy_data.get("aggregate", {})
        lines.append("StrategyQA main results")
        lines.append(f"  EM raw: {agg.get('em_raw', 0.0):.4f}")
        lines.append(f"  EM greedy: {agg.get('em_greedy', 0.0):.4f}")
        lines.append(f"  EM FALCON: {agg.get('em_falcon', 0.0):.4f}")
        lines.append(f"  EM self-reflect: {agg.get('em_self_reflect', 0.0):.4f}")
        lines.append(f"  Avg contradictions before: {agg.get('avg_contradictions_before', 0.0):.4f}")
        lines.append(f"  Avg contradictions after FALCON: {agg.get('avg_contradictions_after_falcon', 0.0):.4f}")
        lines.append("")

    if truthful_data is not None:
        agg = truthful_data.get("aggregate", {})
        lines.append("TruthfulQA main results")
        lines.append(f"  EM raw: {agg.get('em_raw', 0.0):.4f}")
        lines.append(f"  EM greedy: {agg.get('em_greedy', 0.0):.4f}")
        lines.append(f"  EM FALCON: {agg.get('em_falcon', 0.0):.4f}")
        lines.append(f"  EM self-reflect: {agg.get('em_self_reflect', 0.0):.4f}")
        lines.append(f"  Token-F1 FALCON: {agg.get('token_f1_falcon', 0.0):.4f}")
        lines.append(f"  ROUGE-L FALCON: {agg.get('rougeL_falcon', 0.0):.4f}")
        lines.append(f"  Avg contradictions before: {agg.get('avg_contradictions_before', 0.0):.4f}")
        lines.append(f"  Avg contradictions after FALCON: {agg.get('avg_contradictions_after_falcon', 0.0):.4f}")
        lines.append("")

    if experiment_data is not None:
        summary = experiment_data.get("summary", {})
        if summary:
            lines.append("Variant summary")
            for name, stats in summary.items():
                lines.append(
                    f"  {name}: EM FALCON={float(stats.get('em_falcon', 0.0)):.4f}, "
                    f"EM raw={float(stats.get('em_raw', 0.0)):.4f}, "
                    f"Contradictions before={float(stats.get('avg_contradictions_before', 0.0)):.4f}"
                )
            lines.append("")

    if ablation_data is not None:
        runs = ablation_data.get("runs", [])
        if runs:
            best = max(runs, key=lambda r: float(r["summary"].get("em_falcon", 0.0)))
            lines.append("Best ablation configuration")
            lines.append(f"  Name: {best['name']}")
            lines.append(f"  Mode: {best['mode']}")
            lines.append(f"  Tau: {best['tau']}")
            lines.append(f"  Lambda: {best['lambda_penalty']}")
            lines.append(f"  EM FALCON: {float(best['summary'].get('em_falcon', 0.0)):.4f}")
            lines.append("")

    (outdir / "chart_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None, help="Path to strategyqa_main.json")
    parser.add_argument("--truthful", type=str, default=None, help="Path to truthfulqa_main.json")
    parser.add_argument("--experiments", type=str, default=None, help="Path to experiment_results.json")
    parser.add_argument("--ablations", type=str, default=None, help="Path to ablation_results.json")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save charts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_outdir(args.outdir)

    strategy_data = load_json(args.strategy) if args.strategy else None
    truthful_data = load_json(args.truthful) if args.truthful else None
    experiment_data = load_json(args.experiments) if args.experiments else None
    ablation_data = load_json(args.ablations) if args.ablations else None

    chart_main_results(strategy_data, truthful_data, outdir)

    if strategy_data is not None:
        chart_claim_histogram_from_examples("strategyqa", strategy_data, outdir)
    if truthful_data is not None:
        chart_claim_histogram_from_examples("truthfulqa", truthful_data, outdir)

    if experiment_data is not None:
        chart_variant_results(experiment_data, outdir)

    if ablation_data is not None:
        chart_ablation_results(ablation_data, outdir)

    write_summary_text(strategy_data, truthful_data, experiment_data, ablation_data, outdir)

    print(f"Saved charts to: {outdir.resolve()}")


if __name__ == "__main__":
    main()