from __future__ import annotations

import logging
import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def normalize_weight(score: float, default: float = 1.0) -> float:
    """Normalizes a raw score/logit into a positive weight."""
    if score is None:
        return default
    return float(np.exp(score)) if score < 0 else float(score)


class NLIJudge:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base",
                 device: str = "auto", batch_size: int = 8):

        # --- FIX: Resolve 'auto' to actual device ---
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"  # Fast acceleration for Mac
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)

        logger.info("Loading NLI model: %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def contradiction_probs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Computes the probability that pairs[i] is a contradiction.
        Returns a list of floats in [0, 1].
        """
        if not pairs:
            return []

        probs = []
        # Batch processing
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i: i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                # Label 0 is usually contradiction for MNLI models
                # Softmax across the 3 classes (Contra, Entail, Neutral)
                batch_probs = torch.softmax(logits, dim=1)

                # Extract probability of contradiction (index 0)
                contradiction_scores = batch_probs[:, 0].cpu().tolist()
                probs.extend(contradiction_scores)

        return probs


def load_nli_judge(model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "auto") -> NLIJudge:
    return NLIJudge(model_name=model_name, device=device)