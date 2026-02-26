"""
Qualitative Audit for FALCON: Information Loss & Overconfidence

This script performs qualitative analysis on FALCON outputs to assess:
1. Information loss - what valid information is removed?
2. Overconfidence - does filtering create false certainty?
3. Logical validity - are remaining claims truly consistent?
4. Ethical considerations - bias, fairness, transparency

Outputs:
- Detailed audit reports
- Case studies with annotations
- Ethical risk assessments
"""

import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from falcon.models import NLIJudge
from falcon.pipeline import run_falcon_on_text, extract_claims
from main import load_llm_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AuditCase:
    """Individual case for qualitative analysis."""
    input_text: str
    claims_extracted: List[str]
    claims_selected: List[str]
    claims_removed: List[str]
    contradictions_detected: int
    contradiction_pairs: List[Tuple[int, int, float]]

    # Audit flags
    information_loss: str  # none, minor, moderate, severe
    overconfidence_risk: str  # low, medium, high
    logical_validity: str  # valid, questionable, invalid
    ethical_concerns: List[str]

    # Notes
    analyst_notes: str = ""


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assess_information_loss(
    original_claims: List[str],
    selected_claims: List[str],
    removed_claims: List[str]
) -> str:
    """Assess severity of information loss.

    Returns: 'none', 'minor', 'moderate', 'severe'
    """
    if not removed_claims:
        return "none"

    removal_rate = len(removed_claims) / len(original_claims)

    if removal_rate < 0.2:
        return "minor"
    elif removal_rate < 0.5:
        return "moderate"
    else:
        return "severe"


def assess_overconfidence(
    selected_claims: List[str],
    P: Dict[Tuple[int, int], float],
    tau: float
) -> str:
    """Assess risk of overconfidence.

    High risk if:
    - Very few claims remain
    - Claims were borderline contradictory (near tau)
    - Strong assertions without hedging

    Returns: 'low', 'medium', 'high'
    """
    if len(selected_claims) == 0:
        return "high"  # Empty output is maximally confident

    if len(selected_claims) == 1:
        return "medium"  # Single claim may be overconfident

    # Check for borderline contradictions
    borderline_count = 0
    for (i, j), prob in P.items():
        if abs(prob - tau) < 0.1:  # Within 0.1 of threshold
            borderline_count += 1

    if borderline_count > len(selected_claims):
        return "high"
    elif borderline_count > 0:
        return "medium"

    # Check for hedging language
    hedges = ["may", "might", "could", "possibly", "perhaps", "likely"]
    hedge_count = sum(
        1 for claim in selected_claims
        for hedge in hedges
        if hedge in claim.lower()
    )

    if hedge_count == 0 and len(selected_claims) > 2:
        return "medium"  # No hedging may indicate overconfidence

    return "low"


def assess_logical_validity(
    selected_claims: List[str],
    P: Dict[Tuple[int, int], float],
    selected_indices: List[int],
    tau: float
) -> str:
    """Assess logical validity of selected claims.

    Returns: 'valid', 'questionable', 'invalid'
    """
    # Check for residual contradictions
    selected_set = set(selected_indices)
    residual_contradictions = sum(
        1 for (i, j), prob in P.items()
        if i in selected_set and j in selected_set and prob > tau
    )

    if residual_contradictions > 0:
        return "invalid"

    # Check for near-contradictions
    near_contradictions = sum(
        1 for (i, j), prob in P.items()
        if i in selected_set and j in selected_set and prob > tau * 0.8
    )

    if near_contradictions > 0:
        return "questionable"

    return "valid"


def identify_ethical_concerns(
    original_text: str,
    selected_claims: List[str],
    removed_claims: List[str]
) -> List[str]:
    """Identify potential ethical concerns.

    Returns: List of concern descriptions
    """
    concerns = []

    # Check for selective removal of uncertainty
    uncertainty_markers = ["uncertain", "unclear", "unknown", "may", "might", "possibly"]
    removed_uncertainty = sum(
        1 for claim in removed_claims
        for marker in uncertainty_markers
        if marker in claim.lower()
    )
    selected_uncertainty = sum(
        1 for claim in selected_claims
        for marker in uncertainty_markers
        if marker in claim.lower()
    )

    if removed_uncertainty > selected_uncertainty:
        concerns.append("Uncertainty disproportionately removed")

    # Check for bias in claim selection
    sensitive_topics = ["race", "gender", "religion", "ethnicity", "nationality"]
    sensitive_removed = sum(
        1 for claim in removed_claims
        for topic in sensitive_topics
        if topic in claim.lower()
    )

    if sensitive_removed > 0:
        concerns.append(f"Sensitive topics removed ({sensitive_removed} claims)")

    # Check for context loss
    if len(removed_claims) > len(selected_claims):
        concerns.append("Majority of context removed")

    # Check for negation handling
    negations = ["not", "no", "never", "none"]
    for claim in removed_claims:
        for neg in negations:
            if f" {neg} " in f" {claim.lower()} ":
                concerns.append("Negated claims removed - check for bias")
                break

    return concerns


def run_qualitative_audit(
    test_cases: List[str],
    nli: NLIJudge,
    cfg: dict,
    output_dir: Path
) -> List[AuditCase]:
    """Run qualitative audit on test cases."""
    logger.info(f"Running qualitative audit on {len(test_cases)} cases...")

    audits = []
    solver_cfg = cfg.get("solver", {})
    claim_cfg = cfg.get("claims", {})
    tau = float(solver_cfg.get("tau", 0.7))

    for i, text in enumerate(test_cases):
        logger.info(f"Auditing case {i+1}/{len(test_cases)}")

        # Run FALCON
        output, stats, P, claims, weights = run_falcon_on_text(
            text=text,
            nli=nli,
            solver_cfg=solver_cfg,
            claim_cfg=claim_cfg,
            llm=None,
            rewrite_cfg=cfg.get("rewrite", {}),
        )

        # Extract selected/removed claims
        selected_claims = output.split(". ") if output else []
        removed_claims = [c for c in claims if c not in output]

        # Find selected indices
        selected_indices = [i for i, c in enumerate(claims) if c in output]

        # Find contradiction pairs
        contradiction_pairs = [
            (i, j, prob) for (i, j), prob in P.items() if prob > tau
        ]

        # Perform assessments
        info_loss = assess_information_loss(claims, selected_claims, removed_claims)
        overconf = assess_overconfidence(selected_claims, P, tau)
        logic_valid = assess_logical_validity(selected_claims, P, selected_indices, tau)
        ethical = identify_ethical_concerns(text, selected_claims, removed_claims)

        # Create audit case
        audit = AuditCase(
            input_text=text,
            claims_extracted=claims,
            claims_selected=selected_claims,
            claims_removed=removed_claims,
            contradictions_detected=stats.contradictions_before,
            contradiction_pairs=contradiction_pairs,
            information_loss=info_loss,
            overconfidence_risk=overconf,
            logical_validity=logic_valid,
            ethical_concerns=ethical,
            analyst_notes=""
        )

        audits.append(audit)

    return audits


def generate_audit_report(audits: List[AuditCase], output_dir: Path):
    """Generate comprehensive audit report."""
    logger.info("Generating audit report...")

    report = []
    report.append("=" * 80)
    report.append("FALCON QUALITATIVE AUDIT REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Total cases audited: {len(audits)}")
    report.append("")

    # Summary statistics
    info_loss_counts = {"none": 0, "minor": 0, "moderate": 0, "severe": 0}
    overconf_counts = {"low": 0, "medium": 0, "high": 0}
    logic_counts = {"valid": 0, "questionable": 0, "invalid": 0}
    all_ethical_concerns = []

    for audit in audits:
        info_loss_counts[audit.information_loss] += 1
        overconf_counts[audit.overconfidence_risk] += 1
        logic_counts[audit.logical_validity] += 1
        all_ethical_concerns.extend(audit.ethical_concerns)

    # Information Loss
    report.append("--- INFORMATION LOSS ---")
    for level, count in info_loss_counts.items():
        pct = 100 * count / len(audits)
        report.append(f"  {level.capitalize():<12}: {count:>3} ({pct:>5.1f}%)")
    report.append("")

    # Overconfidence Risk
    report.append("--- OVERCONFIDENCE RISK ---")
    for level, count in overconf_counts.items():
        pct = 100 * count / len(audits)
        report.append(f"  {level.capitalize():<12}: {count:>3} ({pct:>5.1f}%)")
    report.append("")

    # Logical Validity
    report.append("--- LOGICAL VALIDITY ---")
    for level, count in logic_counts.items():
        pct = 100 * count / len(audits)
        report.append(f"  {level.capitalize():<12}: {count:>3} ({pct:>5.1f}%)")
    report.append("")

    # Ethical Concerns
    report.append("--- ETHICAL CONCERNS ---")
    if all_ethical_concerns:
        from collections import Counter
        concern_counts = Counter(all_ethical_concerns)
        for concern, count in concern_counts.most_common():
            report.append(f"  • {concern}: {count} cases")
    else:
        report.append("  No ethical concerns identified")
    report.append("")

    # Detailed Cases
    report.append("=" * 80)
    report.append("DETAILED CASE STUDIES")
    report.append("=" * 80)
    report.append("")

    for i, audit in enumerate(audits, 1):
        report.append(f"--- Case {i} ---")
        report.append(f"Input: {audit.input_text[:100]}...")
        report.append(f"Claims extracted: {len(audit.claims_extracted)}")
        report.append(f"Claims selected: {len(audit.claims_selected)}")
        report.append(f"Claims removed: {len(audit.claims_removed)}")
        report.append(f"Contradictions: {audit.contradictions_detected}")
        report.append(f"Information loss: {audit.information_loss}")
        report.append(f"Overconfidence risk: {audit.overconfidence_risk}")
        report.append(f"Logical validity: {audit.logical_validity}")
        if audit.ethical_concerns:
            report.append(f"Ethical concerns: {', '.join(audit.ethical_concerns)}")
        report.append("")

    report.append("=" * 80)
    report.append("RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")

    # Generate recommendations
    severe_loss = info_loss_counts["severe"] + info_loss_counts["moderate"]
    if severe_loss > len(audits) * 0.3:
        report.append("⚠️  HIGH information loss detected in >30% of cases")
        report.append("   → Consider lowering tau threshold")
        report.append("   → Enable soft mode for penalty-based selection")
        report.append("")

    if overconf_counts["high"] > len(audits) * 0.2:
        report.append("⚠️  HIGH overconfidence risk in >20% of cases")
        report.append("   → Add confidence calibration")
        report.append("   → Include uncertainty markers in output")
        report.append("")

    if logic_counts["invalid"] > 0:
        report.append("❌ INVALID logical outputs detected!")
        report.append("   → Review solver constraints")
        report.append("   → Check NLI threshold accuracy")
        report.append("")

    if all_ethical_concerns:
        report.append("⚠️  Ethical concerns identified")
        report.append("   → Review claim selection bias")
        report.append("   → Audit sensitive topic handling")
        report.append("   → Add transparency in filtering rationale")
        report.append("")

    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    report_file = output_dir / "qualitative_audit_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(report_text)
    logger.info(f"Audit report saved to {report_file}")

    # Save detailed JSON
    json_file = output_dir / "qualitative_audit_details.json"
    audit_dicts = [
        {
            "input_text": a.input_text,
            "claims_extracted": a.claims_extracted,
            "claims_selected": a.claims_selected,
            "claims_removed": a.claims_removed,
            "contradictions_detected": a.contradictions_detected,
            "information_loss": a.information_loss,
            "overconfidence_risk": a.overconfidence_risk,
            "logical_validity": a.logical_validity,
            "ethical_concerns": a.ethical_concerns,
        }
        for a in audits
    ]
    with open(json_file, "w") as f:
        json.dump(audit_dicts, f, indent=2)

    logger.info(f"Detailed audit data saved to {json_file}")


def main():
    """Run qualitative audit."""
    logger.info("Starting FALCON qualitative audit...")

    # Setup
    cfg = load_config()
    output_dir = Path("outputs/audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load NLI
    nli_cfg = cfg.get("nli", {})
    nli = NLIJudge(
        model_name=nli_cfg.get("model_name", "cross-encoder/nli-deberta-v3-base"),
        device=nli_cfg.get("device", "auto"),
        batch_size=int(nli_cfg.get("batch_size", 8)),
    )

    # Test cases with known issues
    test_cases = [
        # Information loss
        "The study found that exercise may reduce stress. However, the sample size was small and results were not statistically significant. Further research is needed.",

        # Overconfidence
        "Chocolate is delicious. Dark chocolate contains antioxidants. Milk chocolate has more sugar.",

        # Subtle contradictions
        "Climate change is happening. Some regions are getting warmer. Other regions are experiencing record cold.",

        # Ethical sensitivity
        "People from different cultures have different customs. Some cultures value individualism. Collectivist cultures prioritize group harmony.",

        # Uncertainty handling
        "Scientists believe dark matter exists. Dark matter has not been directly observed. The evidence is indirect and debated.",

        # Negation
        "Vaccines are safe. Vaccines do not cause autism. This has been proven by multiple studies.",

        # Complex reasoning
        "Renewable energy is growing. Solar and wind are cost-competitive. However, storage technology remains a challenge. Grid infrastructure needs upgrading.",
    ]

    # Run audit
    audits = run_qualitative_audit(test_cases, nli, cfg, output_dir)

    # Generate report
    generate_audit_report(audits, output_dir)

    logger.info(f"✅ Qualitative audit complete! Results in {output_dir}")


if __name__ == "__main__":
    main()

