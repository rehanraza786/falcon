"""
Test suite for FALCON improvements:
1. StrategyQA yes/no normalization
2. TruthfulQA free-form prompting
3. Ablation studies
4. Qualitative audits
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_yes_no_extraction():
    """Test StrategyQA yes/no extraction."""
    from falcon.pipeline import extract_yes_no

    logger.info("Testing yes/no extraction...")

    test_cases = [
        ("Yes", "yes"),
        ("No", "no"),
        ("Yes, because of XYZ reason.", "yes"),
        ("No, this is incorrect.", "no"),
        ("The answer is yes.", "yes"),
        ("The answer is no.", "no"),
        ("It is true that...", "yes"),
        ("This is false.", "no"),
        ("Affirmative", "yes"),
        ("Negative", "no"),
    ]

    passed = 0
    for text, expected in test_cases:
        result = extract_yes_no(text)
        if result == expected:
            passed += 1
            logger.info(f"  ✓ '{text[:30]}...' → '{result}'")
        else:
            logger.error(f"  ✗ '{text[:30]}...' → '{result}' (expected '{expected}')")

    logger.info(f"Yes/No extraction: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_truthfulqa_prompting():
    """Test TruthfulQA adapter uses free-form prompts."""
    from falcon.adapters import TruthfulQAAdapter

    logger.info("Testing TruthfulQA prompting...")

    adapter = TruthfulQAAdapter(split="validation")
    test_ex = {"question": "What happens if you eat watermelon seeds?"}

    prompt = adapter.get_question(test_ex)

    # Should NOT contain yes/no constraint
    if "yes or no" in prompt.lower():
        logger.error("  ✗ Prompt still contains 'yes or no' constraint")
        return False

    # Should contain the question
    if "watermelon seeds" not in prompt.lower():
        logger.error("  ✗ Prompt missing question content")
        return False

    logger.info(f"  ✓ Prompt: '{prompt}'")
    logger.info("  ✓ Free-form prompting enabled")
    return True


def test_claim_extraction():
    """Test claim extraction with different settings."""
    from falcon.pipeline import extract_claims

    logger.info("Testing claim extraction...")

    text = "The sky is blue. Water is wet and ice is cold. Birds can fly but penguins cannot."

    # Test with conjunctions
    claims_with = extract_claims(text, split_on_conjunctions=True, max_claims=10)
    logger.info(f"  With conjunctions: {len(claims_with)} claims")

    # Test without conjunctions
    claims_without = extract_claims(text, split_on_conjunctions=False, max_claims=10)
    logger.info(f"  Without conjunctions: {len(claims_without)} claims")

    # With conjunctions should extract more claims
    if len(claims_with) > len(claims_without):
        logger.info("  ✓ Conjunction splitting works")
        return True
    else:
        logger.warning("  ⚠ Conjunction splitting may not be working as expected")
        return False


def test_solver_modes():
    """Test hard vs soft solver modes."""
    from falcon.models import NLIJudge
    from falcon.pipeline import run_falcon_on_text

    logger.info("Testing solver modes...")

    # Setup
    nli = NLIJudge(model_name="cross-encoder/nli-deberta-v3-base", device="cpu")
    text = "The sky is blue. The sky is red. Water is wet."

    claim_cfg = {"split_on_conjunctions": True, "max_claims": 10}

    # Hard mode
    solver_cfg_hard = {"mode": "hard", "tau": 0.7}
    output_hard, stats_hard, _, _, _ = run_falcon_on_text(
        text, nli, solver_cfg_hard, claim_cfg, None, {}
    )

    # Soft mode
    solver_cfg_soft = {"mode": "soft", "tau": 0.7, "lambda_penalty": 1.0}
    output_soft, stats_soft, _, _, _ = run_falcon_on_text(
        text, nli, solver_cfg_soft, claim_cfg, None, {}
    )

    logger.info(f"  Hard mode: {stats_hard.n_claims} claims, {stats_hard.contradictions_after} contradictions")
    logger.info(f"  Soft mode: {stats_soft.n_claims} claims, {stats_soft.contradictions_after} contradictions")

    # Both should run without errors
    logger.info("  ✓ Both solver modes executed successfully")
    return True


def test_config_files():
    """Test config files are valid."""
    import yaml

    logger.info("Testing config files...")

    configs = [
        "config.yaml",
        "config_truthfulqa.yaml",
    ]

    for config_file in configs:
        path = Path(config_file)
        if not path.exists():
            logger.error(f"  ✗ Config not found: {config_file}")
            return False

        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            logger.info(f"  ✓ {config_file} is valid YAML")
        except Exception as e:
            logger.error(f"  ✗ {config_file} failed to parse: {e}")
            return False

    return True


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("FALCON Improvements Test Suite")
    logger.info("=" * 70)
    logger.info("")

    tests = [
        ("Yes/No Extraction", test_yes_no_extraction),
        ("TruthfulQA Prompting", test_truthfulqa_prompting),
        ("Claim Extraction", test_claim_extraction),
        ("Solver Modes", test_solver_modes),
        ("Config Files", test_config_files),
    ]

    results = []
    for name, test_func in tests:
        logger.info("")
        logger.info(f"Running: {name}")
        logger.info("-" * 70)
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append((name, False))

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status:<10} {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    logger.info("")
    logger.info(f"Total: {passed_count}/{total} tests passed")

    if passed_count == total:
        logger.info("✅ All tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

