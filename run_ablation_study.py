from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from falcon.models import NLIJudge
from falcon.pipeline import run_eval
from falcon.utils import set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to a dictionary.")
    return cfg


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping/dictionary.")

    llm = cfg.get("llm", {}) or {}
    if llm.get("enabled", False):
        provider = (llm.get("provider") or "").strip().lower()
        if not provider:
            raise ValueError("llm.enabled is true but llm.provider is missing.")
        if provider not in {"openai", "anthropic", "hf", "vllm_http"}:
            raise ValueError(f"Unsupported llm.provider: {provider}")
        if provider not in cfg:
            raise ValueError(f"Missing provider block '{provider}:' in config.")

    solver = cfg.get("solver", {}) or {}
    mode = (solver.get("mode") or "hard").strip().lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("solver.mode must be 'hard' or 'soft'.")

    eval_cfg = cfg.get("eval", {}) or {}
    ds = (eval_cfg.get("dataset") or "truthfulqa").strip().lower()
    if "strategy" in ds:
        split = (eval_cfg.get("split") or "test").strip().lower()
        if split != "test":
            raise ValueError("StrategyQA adapter supports only split='test'.")


def load_llm_from_config(cfg: dict):
    llm_switch = cfg.get("llm", {}) or {}
    if not llm_switch.get("enabled", False):
        return None

    provider = (llm_switch.get("provider") or "").strip().lower()
    if not provider:
        raise ValueError("llm.enabled is true but llm.provider is missing.")

    provider_cfg = cfg.get(provider, {}) or {}
    if not provider_cfg:
        raise ValueError(f"Missing provider config block: '{provider}:' in config")

    if provider == "openai":
        from falcon.adapters.openai_adapter import OpenAIAdapter, OpenAIConfig
        return OpenAIAdapter(OpenAIConfig(
            model=provider_cfg["model"],
            api_key=provider_cfg.get("api_key"),
            base_url=provider_cfg.get("base_url"),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            max_tokens=int(provider_cfg.get("max_tokens", 256)),
            request_logprobs=bool(provider_cfg.get("request_logprobs", False)),
            top_logprobs=int(provider_cfg.get("top_logprobs", 0)),
        ))

    if provider == "vllm_http":
        from falcon.adapters.vllm_http_adapter import VLLMHTTPAdapter, VLLMHTTPConfig
        return VLLMHTTPAdapter(VLLMHTTPConfig(
            base_url=provider_cfg["base_url"],
            model=provider_cfg["model"],
            api_key=provider_cfg.get("api_key"),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            max_tokens=int(provider_cfg.get("max_tokens", 256)),
            request_logprobs=bool(provider_cfg.get("request_logprobs", False)),
            top_logprobs=int(provider_cfg.get("top_logprobs", 0)),
        ))

    if provider == "hf":
        from falcon.adapters.hf_transformers_adapter import HFTransformersAdapter, HFConfig
        return HFTransformersAdapter(HFConfig(
            model_id=provider_cfg["model_id"],
            device=provider_cfg.get("device", "auto"),
            dtype=provider_cfg.get("dtype", "auto"),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            max_new_tokens=int(provider_cfg.get("max_new_tokens", 256)),
            do_sample=bool(provider_cfg.get("do_sample", True)),
            enable_scoring=bool(provider_cfg.get("enable_scoring", True)),
        ))

    if provider == "anthropic":
        from falcon.adapters.anthropic_adapter import AnthropicAdapter, AnthropicConfig
        return AnthropicAdapter(AnthropicConfig(
            model=provider_cfg["model"],
            api_key=provider_cfg.get("api_key"),
            base_url=provider_cfg.get("base_url"),
            temperature=float(provider_cfg.get("temperature", 0.7)),
            max_tokens=int(provider_cfg.get("max_tokens", 256)),
        ))

    raise ValueError(f"Unknown llm.provider: {provider}")


def make_nli(cfg: Dict[str, Any]) -> NLIJudge:
    nli_cfg = cfg.get("nli", {}) or {}
    return NLIJudge(
        model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-base"),
        device=nli_cfg.get("device", "auto"),
        batch_size=int(nli_cfg.get("batch_size", 8)),
    )


def build_provenance(
    cfg: Dict[str, Any],
    config_path: str,
    seed: int | None,
    ablation_name: str,
) -> Dict[str, Any]:
    llm_switch = cfg.get("llm", {}) or {}
    provider = (llm_switch.get("provider") or "").strip().lower()
    provider_cfg = cfg.get(provider, {}) or {}
    solver_cfg = cfg.get("solver", {}) or {}

    return {
        "config_path": config_path,
        "seed": seed,
        "ablation_name": ablation_name,
        "llm_enabled": bool(llm_switch.get("enabled", False)),
        "llm_provider": provider or None,
        "llm_model": provider_cfg.get("model") or provider_cfg.get("model_id"),
        "nli_model": (cfg.get("nli", {}) or {}).get("model_name"),
        "solver_mode": solver_cfg.get("mode", "hard"),
        "tau": float(solver_cfg.get("tau", 0.7)),
        "lambda_penalty": float(solver_cfg.get("lambda_penalty", 1.0)),
        "weight_source": solver_cfg.get("weight_source", "auto"),
        "rewrite_enabled": bool((cfg.get("rewrite", {}) or {}).get("enabled", False)),
        "self_reflect_enabled": bool((cfg.get("self_reflect", {}) or {}).get("enabled", False)),
        "dataset": (cfg.get("eval", {}) or {}).get("dataset"),
        "split": (cfg.get("eval", {}) or {}).get("split"),
        "max_examples": int(((cfg.get("eval", {}) or {}).get("max_examples", 50))),
    }


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    agg = result.get("aggregate", {}) or {}
    prov = result.get("provenance", {}) or {}
    return {
        "solver_mode": agg.get("solver_mode", prov.get("solver_mode")),
        "tau": agg.get("tau", prov.get("tau")),
        "lambda_penalty": agg.get("lambda_penalty", prov.get("lambda_penalty")),
        "em_raw": agg.get("em_raw"),
        "em_greedy": agg.get("em_greedy"),
        "em_falcon": agg.get("em_falcon"),
        "em_self_reflect": agg.get("em_self_reflect"),
        "token_f1_raw": agg.get("token_f1_raw"),
        "token_f1_greedy": agg.get("token_f1_greedy"),
        "token_f1_falcon": agg.get("token_f1_falcon"),
        "token_f1_self_reflect": agg.get("token_f1_self_reflect"),
        "rougeL_raw": agg.get("rougeL_raw"),
        "rougeL_greedy": agg.get("rougeL_greedy"),
        "rougeL_falcon": agg.get("rougeL_falcon"),
        "rougeL_self_reflect": agg.get("rougeL_self_reflect"),
        "avg_contradictions_before": agg.get("avg_contradictions_before"),
        "avg_contradictions_after_greedy": agg.get("avg_contradictions_after_greedy"),
        "avg_contradictions_after_falcon": agg.get("avg_contradictions_after_falcon"),
        "rewrite_rate": agg.get("rewrite_rate"),
        "self_reflect_rate": agg.get("self_reflect_rate"),
        "avg_latency_s": agg.get("avg_latency_s"),
        "avg_solve_s": agg.get("avg_solve_s"),
    }


def run_one(
    cfg: Dict[str, Any],
    config_path: str,
    seed: int | None,
    ablation_name: str,
) -> Dict[str, Any]:
    validate_config(cfg)
    set_seed(seed)

    nli = make_nli(cfg)
    llm = load_llm_from_config(cfg)
    eval_cfg = cfg.get("eval", {}) or {}

    return run_eval(
        nli=nli,
        solver_cfg=cfg.get("solver", {}) or {},
        claim_cfg=cfg.get("claims", {}) or {},
        dataset_name=eval_cfg.get("dataset", "truthfulqa"),
        split=eval_cfg.get("split", "validation"),
        max_examples=int(eval_cfg.get("max_examples", 50)),
        em_normalize=bool(eval_cfg.get("em_normalize", True)),
        llm=llm,
        rewrite_cfg=cfg.get("rewrite", {}) or {},
        self_reflect_cfg=cfg.get("self_reflect", {}) or {},
        provenance=build_provenance(cfg, config_path, seed, ablation_name),
    )


def parse_float_list(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="results/ablation_results.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--taus", nargs="*", default=["0.5", "0.6", "0.7", "0.8", "0.9"])
    parser.add_argument("--lambdas", nargs="*", default=["0.5", "1.0", "1.5", "2.0"])
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["hard", "soft"],
        help="Subset of solver modes to test",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    taus = parse_float_list(args.taus)
    lambdas = parse_float_list(args.lambdas)

    all_results: Dict[str, Any] = {
        "config_path": args.config,
        "seed": args.seed,
        "grid": {
            "taus": taus,
            "lambdas": lambdas,
            "modes": args.modes,
        },
        "runs": [],
        "summary": [],
    }

    for mode in args.modes:
        if mode not in {"hard", "soft"}:
            raise ValueError(f"Unsupported mode '{mode}'")

        for tau, lam in itertools.product(taus, lambdas):
            cfg = deepcopy(base_cfg)
            cfg.setdefault("solver", {})["mode"] = mode
            cfg["solver"]["tau"] = float(tau)
            cfg["solver"]["lambda_penalty"] = float(lam)

            ablation_name = f"{mode}_tau{tau}_lambda{lam}"
            result = run_one(cfg, args.config, args.seed, ablation_name)
            summary = summarize_result(result)

            run_record = {
                "name": ablation_name,
                "tau": float(tau),
                "lambda_penalty": float(lam),
                "mode": mode,
                "result": result,
                "summary": summary,
            }
            all_results["runs"].append(run_record)

            compact = {
                "name": ablation_name,
                "tau": tau,
                "lambda_penalty": lam,
                "mode": mode,
            }
            compact.update(summary)
            all_results["summary"].append(compact)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved ablation study results to {out_path}")
    print(json.dumps(all_results["summary"], indent=2))


if __name__ == "__main__":
    main()