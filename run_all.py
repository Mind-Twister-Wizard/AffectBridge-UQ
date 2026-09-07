from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_manifest import build_manifest
from src.config import load_config
from src.demo import make_demo_manifest
from src.download_data import download_datasets
from src.features import create_demo_cache, extract_wavlm_cache
from src.reporting import write_report
from src.train import run_experiments
from src.utils import device_from_arg, ensure_dir, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click AffectBridge-UQ v2 paper pipeline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--download", action="store_true", help="Download public Kaggle corpora.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--train", action="store_true", help="Run model training and evaluation.")
    parser.add_argument("--report", action="store_true", help="Create HTML report and publication figures.")
    parser.add_argument("--demo", action="store_true", help="Tiny synthetic end-to-end validation; no downloads.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", nargs="*", default=None, help="Optional held-out corpora.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override paper seeds, e.g. --seeds 42 52 62")
    parser.add_argument("--ablation-scope", choices=["none", "representative", "full"], default="representative")
    args = parser.parse_args()

    if not any((args.download, args.train, args.report, args.demo)):
        args.download = args.train = args.report = True

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    set_seed(int(cfg["seed"]))
    device = device_from_arg(args.device)
    output_root = ensure_dir(cfg["output_dir"])
    cuda_available = bool(torch.cuda.is_available())
    print(f"AffectBridge-UQ v2 | device={device} | root={PROJECT_ROOT}")
    print(f"PyTorch {torch.__version__} | CUDA available={cuda_available}")
    if cuda_available:
        print(f"CUDA device={torch.cuda.get_device_name(0)}")
    elif shutil.which("nvidia-smi"):
        print("WARNING: nvidia-smi is available but this PyTorch build cannot see CUDA; use the launcher so it can install the CUDA wheel.")

    if args.demo:
        manifest = make_demo_manifest(cfg)
        ensure_dir(cfg["metadata_dir"])
        manifest.to_csv(Path(cfg["metadata_dir"]) / "manifest.csv", index=False)
        feature_cache = create_demo_cache(cfg, manifest, int(cfg["seed"]))
        cfg["demo"] = True
        cfg["resume_completed"] = False
        if args.epochs is None:
            cfg["epochs"] = 3
        if args.folds is None:
            args.folds = list(cfg["kaggle_datasets"].keys())
        if args.seeds is None:
            args.seeds = [int(cfg["seed"])]
    else:
        if args.download or not (Path(cfg["metadata_dir"]) / "manifest.csv").exists():
            download_datasets(cfg, force=args.force_download)
        manifest = build_manifest(cfg)
        feature_cache = extract_wavlm_cache(cfg, manifest, device, force=args.force_cache)

    save_json({
        "config": cfg,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }, output_root / "run_context.json")

    if args.demo or args.train or not (output_root / "all_results.json").exists():
        records = run_experiments(
            cfg,
            manifest,
            feature_cache,
            device,
            output_root,
            folds=args.folds,
            ablation_scope=args.ablation_scope,
            max_epochs=args.epochs,
            seeds=args.seeds,
        )
    else:
        records = json.loads((output_root / "all_results.json").read_text(encoding="utf-8"))

    if args.report:
        report = write_report(cfg, manifest, records, output_root)
        print(f"Report: {report}")
    print(f"Artifacts: {output_root}")


if __name__ == "__main__":
    main()
