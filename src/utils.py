from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
try:  # Keep download/manifest commands usable before the ML stack is imported.
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None  # type: ignore[assignment]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(paths: list[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(Path(p)) for p in paths):
        p = Path(path)
        digest.update(str(p).encode())
        digest.update(str(p.stat().st_size).encode())
        digest.update(str(p.stat().st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def device_from_arg(name: str | None) -> torch.device:
    if torch is None:
        raise RuntimeError("PyTorch is required for training. Install requirements.txt first.")
    if name and name.lower() != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
