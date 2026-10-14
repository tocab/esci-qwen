"""Shared configuration and reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LABELS = ("E", "S", "C", "I")
CRITERIA = {
    "E": "Exact match: directly satisfies the query intent",
    "S": "Substitute: an alternative satisfying a similar need",
    "C": "Complement: accessory or related item used with the searched item",
    "I": "Irrelevant: does not satisfy the query intent",
}


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read YAML configuration and validate its four-label prompt order."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if len(config["prompt"]["order"]) != 4 or set(config["prompt"]["order"]) != set(LABELS):
        raise ValueError("Prompt order must contain each ESCI label exactly once")
    return config


def write_json(path: str | os.PathLike[str], obj: Any) -> None:
    """Write a finite JSON payload through a temporary file and atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: str | os.PathLike[str]) -> str:
    """Compute a file's SHA-256 digest without loading it entirely into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels where available."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # DeltaNet kernels may not have a deterministic implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment() -> dict[str, Any]:
    """Describe the Python, package, CUDA, and GPU versions used by the current run."""
    import importlib.metadata
    import platform

    result = {
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "numpy", "scikit-learn")
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for package in ("fla-core", "causal-conv1d", "triton", "einops"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    return result
