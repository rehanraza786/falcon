import argparse
import json
import logging
import sys
import yaml
from pathlib import Path

from falcon.models import NLIJudge
from falcon.pipeline import run_falcon_on_text, run_eval

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to a dictionary.")
    return cfg


def validate_config(cfg: dict) -> None:
    """Lightweight config validation with actionable errors."""
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
            raise ValueError(f"Missing provider block '{provider}:' in config.yaml")

    solver = cfg.get("solver", {}) or {}
    mode = (solver.get("mode") or "hard").strip().lower()
    if mode not in {"hard", "soft"}:
        raise ValueError("solver.mode must be 'hard' or 'soft'.")

    eval_cfg = cfg.get("eval", {}) or {}
    ds = (eval_cfg.get("dataset") or "truthfulqa").strip().lower()
    if "strategy" in ds:
        # Adapter only supports test split.
        split = (eval_cfg.get("split") or "test").strip().lower()
        if split != "test":
            raise ValueError("StrategyQA adapter supports only split='test'. Update eval.split in config.yaml.")

def load_llm_from_config(cfg: dict):
    """Create and return an LLM adapter based on config.

    We keep `llm:` as a small on/off switch + provider selector, while
    provider-specific settings live under top-level blocks like `openai:`,
    `anthropic:`, `hf:`, `vllm_http:`.

    This matches `config.yaml` and makes it easy to swap providers.
    """
    llm_switch = cfg.get("llm", {}) or {}
    if not llm_switch.get("enabled", False):
        return None

    provider = (llm_switch.get("provider") or "").strip().lower()
    if not provider:
        raise ValueError("llm.enabled is true but llm.provider is missing.")

    provider_cfg = cfg.get(provider, {}) or {}
    if not provider_cfg:
        raise ValueError(f"Missing provider config block: '{provider}:' in config.yaml")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "eval"], required=True)
    parser.add_argument("--text", help="Text for single mode")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)")
    parser.add_argument("--log-file", default=None, help="Optional path to write logs")
    parser.add_argument("--logic", choices=["hard", "soft"], help="Override solver mode")
    parser.add_argument("--out", help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_config(cfg)

    # 1) Setup NLI
    nli_cfg = cfg.get("nli", {}) or {}
    nli = NLIJudge(
        model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-base"),
        device=nli_cfg.get("device", "auto"),
        batch_size=int(nli_cfg.get("batch_size", 8)),
    )

    # 2) Setup LLM (optional)
    llm = load_llm_from_config(cfg)

    # 3) Solver config
    solver_cfg = cfg.get("solver", {}) or {}
    if args.logic:
        solver_cfg["mode"] = args.logic

    if args.mode == "single":
        if not args.text:
            logger.error("--text required for single mode")
            sys.exit(1)

        output, stats, P, claims, weights = run_falcon_on_text(
            text=args.text,
            nli=nli,
            solver_cfg=solver_cfg,
            claim_cfg=cfg.get("claims", {}) or {},
            llm=llm,
            rewrite_cfg=cfg.get("rewrite", {}) or {},
        )

        logger.info("Input: %s", args.text)
        logger.info("FALCON output: %s", output)
        logger.info("Stats: Removed %d contradictions.", stats.contradictions_before - stats.contradictions_after)

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(
                    {"input": args.text, "output": output, "stats": stats.__dict__},
                    f,
                    indent=2,
                )

    else:  # eval
        eval_cfg = cfg.get("eval", {}) or {}
        dataset_name = eval_cfg.get("dataset", "truthfulqa")
        split = eval_cfg.get("split", "validation")
        max_examples = int(eval_cfg.get("max_examples", 50))
        em_normalize = bool(eval_cfg.get("em_normalize", True))

        logger.info("Starting eval: %s (%s logic)", dataset_name, solver_cfg.get("mode", "hard"))
        results = run_eval(
            nli=nli,
            solver_cfg=solver_cfg,
            claim_cfg=cfg.get("claims", {}) or {},
            dataset_name=dataset_name,
            split=split,
            max_examples=max_examples,
            em_normalize=em_normalize,
            llm=llm,
            rewrite_cfg=cfg.get("rewrite", {}) or {},
        )

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
