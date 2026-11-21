"""holoveil training and PoC verification package (DESIGN.md v0.5.2-design)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__version__ = "0.5.2"

DESIGN_VERSION = "0.5.2-design"
DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


__all__ = [
    "__version__",
    "DESIGN_VERSION",
    "DEFAULT_SEED",
    "seed_everything",
]
