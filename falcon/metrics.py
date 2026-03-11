from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []


def extract_yes_no(text: str) -> str:
    text_lower = (text or "").strip().lower()

    if text_lower in ("yes", "no"):
        return text_lower

    if re.search(r"\byes\b", text_lower):
        return "yes"
    if re.search(r"\bno\b", text_lower):
        return "no"

    return normalize_text(text)


def exact_match(pred: str, gold: str, dataset_name: str, em_normalize: bool = True) -> float:
    ds_name = (dataset_name or "").lower()
    if not em_normalize:
        return float((pred or "").strip() == (gold or "").strip())

    if "strategy" in ds_name:
        return float(extract_yes_no(pred) == extract_yes_no(gold))
    return float(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(
    pred: str,
    gold: str,
    dataset_name: str,
    em_normalize: bool = True,
) -> Dict[str, float]:
    return {
        "em": exact_match(pred, gold, dataset_name, em_normalize),
        "token_f1": token_f1(pred, gold),
        "rougeL": rouge_l(pred, gold),
    }