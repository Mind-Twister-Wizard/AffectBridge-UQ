from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    root = path.parent.resolve()
    cfg["root_dir"] = str(root)
    cfg["data_dir"] = str(root / "data")
    cfg["raw_dir"] = str(root / "data" / "raw")
    cfg["metadata_dir"] = str(root / "data" / "metadata")
    cfg["cache_dir"] = str(root / "data" / "cache")
    cfg["output_dir"] = str(root / "outputs")
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg
