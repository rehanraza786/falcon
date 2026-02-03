import os
import sys
import json
import yaml
import copy
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=getattr(logging, os.getenv("FALCON_LOGLEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

# --- Configuration ---
MODES = ["hard", "soft"]
DATASETS = ["truthfulqa", "strategyqa"]

BASE_DIR = Path(__file__).parent.resolve()
BASE_CONFIG = (BASE_DIR / "config.yaml").resolve()
OUTPUT_DIR = (BASE_DIR / os.environ.get("OUTPUT_DIR", "outputs")).resolve()

ENABLE_LLM = os.environ.get("ENABLE_LLM", "0").lower() in ("1", "true", "yes", "y")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "vllm_http")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())


def run_cmd(cmd: list[str]) -> None:
    logger.info("Running: python main.py ... %s", Path(cmd[-1]).name)
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


def summarize_outputs(out_dir: Path) -> None:
    import csv
    rows = []
    for p in sorted(out_dir.glob("*_results.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        stem = p.stem.replace("_results", "")
        parts = stem.rsplit("_", 1)
        dataset, mode = parts if len(parts) == 2 else ("unknown", "unknown")

        agg = obj.get("aggregate", {})
        rows.append({
            "file": p.name,
            "dataset": dataset,
            "logic": mode,
            "em_raw": agg.get("em_raw"),
            "em_greedy": agg.get("em_greedy"),
            "em_falcon": agg.get("em_falcon"),
            "solve_s": agg.get("avg_solve_s"),
            "contra_after_greedy": agg.get("avg_contradictions_after_greedy"),
            "contra_after_falcon": agg.get("avg_contradictions_after_falcon"),
        })

    if rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print(f"\n📊 Summary saved to {out_dir}/summary.csv")


def default_split_for(dataset: str) -> str:
    ds = dataset.lower()
    if ds == "strategyqa":
        return "test"
    if ds == "truthfulqa":
        return "validation"
    return "validation"


def run_experiment() -> None:
    if not BASE_CONFIG.exists():
        print(f"❌ Error: Config not found at {BASE_CONFIG}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(BASE_CONFIG)

    print(f"🚀 Starting Run in {BASE_DIR}")

    for dataset in DATASETS:
        for mode in MODES:
            print(f"▶ {dataset} | {mode}")

            out_file = (OUTPUT_DIR / f"{dataset}_{mode}_results.json").resolve()
            temp_cfg = (OUTPUT_DIR / f"temp_{dataset}_{mode}.yaml").resolve()

            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("eval", {})

            # Always override dataset & split to prevent stale base-config splits
            cfg["eval"]["dataset"] = dataset
            cfg["eval"]["split"] = default_split_for(dataset)

            # Optional LLM enable
            if ENABLE_LLM:
                cfg.setdefault("llm", {})
                cfg["llm"].update({"enabled": True, "provider": LLM_PROVIDER})

            save_yaml(cfg, temp_cfg)

            cmd = [
                sys.executable, str(BASE_DIR / "main.py"),
                "--mode", "eval",
                "--config", str(temp_cfg),
                "--logic", mode,
                "--out", str(out_file)
            ]

            try:
                run_cmd(cmd)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {e}")
            finally:
                if temp_cfg.exists():
                    temp_cfg.unlink()

    summarize_outputs(OUTPUT_DIR)


if __name__ == "__main__":
    run_experiment()
