from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from .config import load_config
from .utils import ensure_dir


def _has_audio(path: Path) -> bool:
    return any(next(path.rglob(ext), None) is not None for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"))


def _folder_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _credential_source() -> str | None:
    """Return the configured Kaggle credential mechanism, without exposing secrets.

    Kaggle supports both the legacy ``kaggle.json`` file and the newer API-token
    file/environment variable.  Checking this before importing either client is
    important: a missing token should produce a useful setup message, not an
    opaque import error from an optional Kaggle dependency.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY"

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    if (config_dir / "access_token").is_file():
        return "~/.kaggle/access_token"
    if (config_dir / "kaggle.json").is_file():
        return "~/.kaggle/kaggle.json"
    return None


def _authentication_message() -> str:
    """Give copy/pasteable credential instructions while keeping secrets out of logs."""
    return (
        "Kaggle authentication is required to download the configured corpora. "
        "Create an API token at https://www.kaggle.com/settings/api, then use one "
        "of these supported methods: place kaggle.json in "
        "<your user folder>/.kaggle/kaggle.json, place the token in "
        "<your user folder>/.kaggle/access_token, or set KAGGLE_API_TOKEN (or "
        "KAGGLE_USERNAME and KAGGLE_KEY). The package never stores credentials."
    )


def _download_with_kagglehub(handle: str, destination: Path) -> Path:
    import kagglehub  # type: ignore

    ensure_dir(destination)
    try:
        resolved = kagglehub.dataset_download(handle, output_dir=str(destination))
    except TypeError:
        resolved = kagglehub.dataset_download(handle, path=None)
    resolved_path = Path(str(resolved))
    if resolved_path.is_file() and resolved_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(resolved_path) as archive:
            archive.extractall(destination)
        return destination
    if resolved_path.exists() and resolved_path.resolve() != destination.resolve():
        if resolved_path.is_dir():
            for item in resolved_path.iterdir():
                target = destination / item.name
                if target.exists():
                    continue
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)
    return destination


def _download_with_kaggle_cli(handle: str, destination: Path) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

    ensure_dir(destination)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(handle, path=str(destination), unzip=True, quiet=False)
    return destination


def download_datasets(cfg: dict[str, Any], only: list[str] | None = None, force: bool = False) -> dict[str, Any]:
    raw_root = ensure_dir(cfg["raw_dir"])
    requested = set(only or cfg["kaggle_datasets"].keys())
    statuses: dict[str, Any] = {}
    for name, item in cfg["kaggle_datasets"].items():
        if name not in requested:
            continue
        destination = raw_root / name
        ensure_dir(destination)
        if force and destination.exists():
            for child in destination.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        if _has_audio(destination) and not force:
            statuses[name] = {"status": "already_present", "path": str(destination)}
            continue
        # KaggleHub can download some public datasets anonymously.  Let that
        # path run first; only require a credential if the resource rejects it.
        credential_source = _credential_source()
        handle = item["handle"]
        errors: list[str] = []
        try:
            _download_with_kagglehub(handle, destination)
        except Exception as exc:  # pragma: no cover - depends on external credentials/network
            errors.append(f"kagglehub: {exc}")
        if not _has_audio(destination) and credential_source is not None:
            try:
                _download_with_kaggle_cli(handle, destination)
            except Exception as cli_exc:
                errors.append(f"kaggle API: {cli_exc}")
        if not _has_audio(destination):
            if credential_source is None:
                errors.append("kaggle API fallback skipped: no credential was detected")
            source_hint = f"Credential source detected: {credential_source}. " if credential_source else ""
            raise RuntimeError(
                f"Could not find audio files for {name}. Tried Kaggle downloads. "
                f"{source_hint}{_authentication_message()} "
                + " | ".join(errors)
            )
        statuses[name] = {
            "status": "downloaded",
            "path": str(destination),
            "handle": handle,
            "credential_source": credential_source or "anonymous KaggleHub",
        }
    total_bytes = sum(_folder_size_bytes(raw_root / name) for name in requested if (raw_root / name).exists())
    limit = float(cfg.get("max_total_dataset_gb", 5.0)) * (1024 ** 3)
    if total_bytes > limit:
        raise RuntimeError(f"Downloaded corpora total {total_bytes / (1024 ** 3):.2f} GiB, exceeding the configured {limit / (1024 ** 3):.2f} GiB limit.")
    (raw_root / "download_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public Kaggle corpora for AffectBridge-UQ.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--only", nargs="*", help="Optional corpus names: crema_d ravdess tess savee")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(json.dumps(download_datasets(cfg, args.only, args.force), indent=2))


if __name__ == "__main__":
    main()
