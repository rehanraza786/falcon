from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int]) -> None:
    """Set RNG seeds for reproducibility.

    If `seed` is None, this is a no-op.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    # Torch is optional in some environments; seed it when present.
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic mode can reduce performance; keep it opt-in via env var.
        if os.getenv("FALCON_TORCH_DETERMINISTIC", "0") == "1":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def setup_logging(verbosity: int = 0, log_file: Optional[str] = None) -> None:
    """Configure root logging.

    verbosity:
      0 -> WARNING
      1 -> INFO
      2+ -> DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
