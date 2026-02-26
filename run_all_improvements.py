"""
Master script to run all FALCON improvements:
1. Fix StrategyQA evaluation with yes/no normalization
2. Fix TruthfulQA with free-form prompting and higher temperature
3. Run ablation studies (tau, max_claims, solver robustness)
4. Conduct qualitative audits
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ {description} failed: {e}")
        return False


def main():
    """Run all improvements."""
    logger.info("=" * 70)
    logger.info("FALCON COMPREHENSIVE IMPROVEMENTS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This script will:")
    logger.info("  1. Test all improvements")
    logger.info("  2. Run StrategyQA evaluation (with yes/no normalization)")
    logger.info("  3. Run TruthfulQA evaluation (free-form, high temp)")
    logger.info("  4. Conduct ablation studies")
    logger.info("  5. Perform qualitative audits")
    logger.info("")

    # Ensure output directories exist
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/ablation").mkdir(exist_ok=True)
    Path("outputs/audit").mkdir(exist_ok=True)

    results = []

    # Step 1: Test improvements
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Testing Improvements")
    logger.info("=" * 70)
    results.append((
        "Test Suite",
        run_command([sys.executable, "test_improvements.py"], "Test Suite")
    ))

    # Step 2: Run StrategyQA evaluation (fixed)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: StrategyQA Evaluation (Yes/No Normalization)")
    logger.info("=" * 70)

    for mode in ["hard", "soft"]:
        results.append((
            f"StrategyQA ({mode})",
            run_command([
                sys.executable, "main.py",
                "--mode", "eval",
                "--config", "config.yaml",
                "--logic", mode,
                "--out", f"outputs/strategyqa_{mode}_fixed.json"
            ], f"StrategyQA Evaluation ({mode} mode)")
        ))

    # Step 3: Run TruthfulQA evaluation (fixed)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: TruthfulQA Evaluation (Free-form, High Temp)")
    logger.info("=" * 70)

    for mode in ["hard", "soft"]:
        results.append((
            f"TruthfulQA ({mode})",
            run_command([
                sys.executable, "main.py",
                "--mode", "eval",
                "--config", "config_truthfulqa.yaml",
                "--logic", mode,
                "--out", f"outputs/truthfulqa_{mode}_fixed.json"
            ], f"TruthfulQA Evaluation ({mode} mode)")
        ))

    # Step 4: Ablation studies
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Ablation Studies")
    logger.info("=" * 70)
    results.append((
        "Ablation Study",
        run_command([sys.executable, "run_ablation_study.py"], "Ablation Study")
    ))

    # Step 5: Qualitative audit
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Qualitative Audit")
    logger.info("=" * 70)
    results.append((
        "Qualitative Audit",
        run_command([sys.executable, "run_qualitative_audit.py"], "Qualitative Audit")
    ))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("")

    for name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{status:<15} {name}")

    total = len(results)
    passed = sum(1 for _, s in results if s)

    logger.info("")
    logger.info(f"Completed: {passed}/{total} tasks successful")
    logger.info("")

    if passed == total:
        logger.info("🎉 All improvements completed successfully!")
        logger.info("")
        logger.info("Results available in:")
        logger.info("  - outputs/strategyqa_*_fixed.json")
        logger.info("  - outputs/truthfulqa_*_fixed.json")
        logger.info("  - outputs/ablation/")
        logger.info("  - outputs/audit/")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} task(s) failed")
        logger.error("Review the logs above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())

