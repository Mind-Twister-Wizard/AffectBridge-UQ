from __future__ import annotations

import argparse
import copy
import json
import math
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment import boundary_mix, same_class_cross_corpus_mix
from .calibrate import fit_classwise_conformal, predict_with_deferral
from .config import load_config
from .dataset import CachedAffectDataset
from .features import FeatureCache
from .losses import compute_loss, uniform_prediction_loss
from .metrics import compute_metrics
from .model import AffectBridgeUQ
from .splits import add_encoded_columns, encode_training_columns, make_lodo_splits
from .utils import device_from_arg, ensure_dir, num_trainable_parameters, save_json, set_seed




def _config_digest(cfg: dict[str, Any], ablation_name: str, ablation: dict[str, Any], holdout: str, seed: int, cache_meta: dict[str, Any] | None = None) -> str:
    payload = {
        "implementation": "DIMAP-C-v2.0",
        "config": cfg,
        "ablation_name": ablation_name,
        "ablation": ablation,
        "holdout": holdout,
        "seed": int(seed),
        "cache_signature": (cache_meta or {}).get("manifest_signature"),
        "feature_layers": (cache_meta or {}).get("feature_layers"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _load_completed_run(output_root: Path, cfg: dict[str, Any], feature_cache: FeatureCache, holdout: str, ablation_name: str, ablation: dict[str, Any], seed: int) -> dict[str, Any] | None:
    if not bool(cfg.get("resume_completed", True)):
        return None
    run_dir = output_root / "runs" / f"{ablation_name}__holdout_{holdout}__seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not (metrics_path.exists() and predictions_path.exists() and checkpoint_path.exists()):
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _, cache_meta = feature_cache.load()
        expected = _config_digest(cfg, ablation_name, ablation, holdout, seed, cache_meta)
        if metrics.get("config_digest") != expected:
            return None
        history_path = run_dir / "history.json"
        calibration_path = run_dir / "calibration.json"
        metrics.update({
            "run_dir": str(run_dir),
            "history": json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else [],
            "calibration": json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {},
            "predictions_path": str(predictions_path),
        })
        print(f"Reusing completed run: {ablation_name} | holdout={holdout} | seed={seed}")
        return metrics
    except Exception:
        return None

ABLATIONS: dict[str, dict[str, Any]] = {
    "full_DIMAP-C": {},
    "source_only_adapter": {
        "use_prototypes": False, "prototype_weight": 0.0,
        "use_domain_adversarial": False, "domain_adversarial_weight": 0.0,
        "use_speaker_adversarial": False, "speaker_adversarial_weight": 0.0,
        "conditional_alignment_weight": 0.0, "cross_corpus_mixup_weight": 0.0, "boundary_entropy_weight": 0.0,
    },
    "DANN_adapter": {
        "use_prototypes": False, "prototype_weight": 0.0,
        "use_speaker_adversarial": False, "speaker_adversarial_weight": 0.0,
        "conditional_alignment_weight": 0.0, "cross_corpus_mixup_weight": 0.0, "boundary_entropy_weight": 0.0,
    },
    "no_prototypes": {"use_prototypes": False, "prototype_weight": 0.0},
    "global_prototype": {"global_prototype": True, "prototype_consistency_weight": 0.0},
    "no_domain_adversarial": {"use_domain_adversarial": False, "domain_adversarial_weight": 0.0},
    "no_speaker_adversarial": {"use_speaker_adversarial": False, "speaker_adversarial_weight": 0.0},
    "no_cross_corpus_mixup": {"cross_corpus_mixup_weight": 0.0, "boundary_entropy_weight": 0.0},
    "no_conformal_deferral": {"use_conformal": False, "use_shift_guard": False},
}


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # pragma: no cover - compatibility with older Torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _balanced_sample_weights(manifest: pd.DataFrame, indices: np.ndarray, power: float) -> torch.Tensor:
    subset = manifest.iloc[indices]
    cell_counts = subset.groupby(["corpus", "label_id"]).size().to_dict()
    values = []
    for _, row in subset.iterrows():
        count = max(1, int(cell_counts[(row["corpus"], int(row["label_id"]))]))
        values.append(count ** (-float(power)))
    weights = np.asarray(values, dtype=np.float64)
    weights /= max(weights.mean(), 1e-12)
    return torch.as_tensor(weights, dtype=torch.double)


def _loader(
    features: np.ndarray,
    manifest: pd.DataFrame,
    indices: np.ndarray,
    cfg: dict[str, Any],
    shuffle: bool,
    device: torch.device,
    seed: int = 42,
    balanced: bool = False,
) -> DataLoader:
    dataset = CachedAffectDataset(features, manifest, indices)
    sampler = None
    if balanced and len(indices):
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        sampler = WeightedRandomSampler(
            _balanced_sample_weights(manifest, indices, float(cfg.get("corpus_class_balance_power", 0.70))),
            num_samples=len(indices),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


@torch.no_grad()
def collect_outputs(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits, dispersions, proto_distances, labels, corpora, indices, paths = [], [], [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        output = model(x)
        logits.append(output["logits"].float().cpu())
        dispersions.append(output["dispersion"].float().cpu())
        proto_distances.append(output["prototype_distance"].float().cpu())
        labels.append(batch["label"].cpu())
        corpora.append(batch["corpus"].cpu())
        indices.append(batch["index"].cpu())
        paths.extend(batch["path"])
    return {
        "logits": torch.cat(logits) if logits else torch.empty((0, model.num_classes)),
        "dispersion": torch.cat(dispersions) if dispersions else torch.empty(0),
        "prototype_distance": torch.cat(proto_distances) if proto_distances else torch.empty(0),
        "labels": torch.cat(labels) if labels else torch.empty(0, dtype=torch.long),
        "corpus": torch.cat(corpora) if corpora else torch.empty(0, dtype=torch.long),
        "indices": torch.cat(indices) if indices else torch.empty(0, dtype=torch.long),
        "paths": paths,
    }


def _validation_scores(outputs: dict[str, Any]) -> tuple[float, float]:
    labels = outputs["labels"].numpy()
    predictions = outputs["logits"].argmax(dim=-1).numpy()
    corpora = outputs["corpus"].numpy()
    keep = labels >= 0
    if not keep.any():
        return -float("inf"), -float("inf")
    pooled = float(f1_score(labels[keep], predictions[keep], average="macro", zero_division=0))
    per_domain = []
    for corpus in np.unique(corpora[keep]):
        mask = keep & (corpora == corpus)
        if mask.any():
            per_domain.append(f1_score(labels[mask], predictions[mask], average="macro", zero_division=0))
    domain_mean = float(np.mean(per_domain)) if per_domain else pooled
    return domain_mean, pooled


def _save_predictions(path: Path, outputs: dict[str, Any], decisions: dict[str, np.ndarray], label_names: list[str]) -> None:
    frame = pd.DataFrame({
        "manifest_index": outputs["indices"].numpy(),
        "true_label_id": outputs["labels"].numpy(),
        "prediction_id": decisions["prediction"],
        "set_size": decisions["set_sizes"],
        "dispersion": decisions["dispersion"],
        "prototype_distance": decisions["prototype_distance"],
        "unknown_score": decisions["unknown_score"],
        "max_probability": decisions["max_probability"],
        "temperature": decisions["temperature"],
        "abstain": decisions["abstain"],
        "decision": decisions["decision"],
    })
    frame["true_label"] = frame["true_label_id"].map(lambda x: label_names[int(x)] if int(x) >= 0 and int(x) < len(label_names) else "unknown")
    frame["prediction"] = frame["prediction_id"].map(lambda x: label_names[int(x)])
    for i, label in enumerate(label_names):
        frame[f"prob_{label}"] = decisions["probabilities"][:, i]
    frame.to_csv(path, index=False)


def _ema_update(ema_model: nn.Module, model: nn.Module, step: int, max_decay: float = 0.995) -> None:
    # Early EMA must not remain dominated by the random initialization.
    # Copy exactly on the first step, then increase decay smoothly.
    if step <= 1:
        ema_model.load_state_dict(model.state_dict())
    else:
        decay = min(float(max_decay), 1.0 - 1.0 / float(step + 1))
        with torch.no_grad():
            ema_params = dict(ema_model.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
            ema_buffers = dict(ema_model.named_buffers())
            for name, buffer in model.named_buffers():
                if name in ema_buffers:
                    ema_buffers[name].copy_(buffer)
    if hasattr(model, "prototype_scale"):
        ema_model.prototype_scale = model.prototype_scale


def _set_epoch_lr(optimizer: AdamW, epoch: int, epochs: int, cfg: dict[str, Any]) -> float:
    base = float(cfg["learning_rate"])
    minimum = float(cfg.get("min_learning_rate", base * 0.05))
    warmup = max(1, int(cfg.get("warmup_epochs", 2)))
    if epoch <= warmup:
        lr = base * epoch / warmup
    else:
        progress = (epoch - warmup - 1) / max(1, epochs - warmup - 1)
        lr = minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def fit_fold(
    cfg: dict[str, Any],
    manifest: pd.DataFrame,
    feature_cache: FeatureCache,
    holdout: str,
    ablation_name: str,
    ablation: dict[str, Any],
    device: torch.device,
    output_root: Path,
    seed: int,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    features, cache_meta = feature_cache.load()
    manifest = manifest.reset_index(drop=True).copy()
    split = make_lodo_splits(manifest, holdout, seed=seed)
    source_corpora = sorted(manifest.iloc[split["train"]]["corpus"].astype(str).unique())
    corpus_map, speaker_map = encode_training_columns(manifest, split["train"], split["val"], source_corpora)
    encoded = add_encoded_columns(manifest, corpus_map, speaker_map, list(cfg["known_labels"]))
    train_loader = _loader(features, encoded, split["train"], cfg, shuffle=False, device=device, seed=seed, balanced=True)
    val_loader = _loader(features, encoded, split["val"], cfg, shuffle=False, device=device, seed=seed, balanced=False)
    test_loader = _loader(features, encoded, split["test"], cfg, shuffle=False, device=device, seed=seed, balanced=False)

    flags = dict(ablation)
    effective_cfg = {**cfg, **flags}
    model = AffectBridgeUQ(
        input_dim=int(cache_meta["feature_dim"]),
        num_classes=len(cfg["known_labels"]),
        num_corpora=len(source_corpora),
        num_speakers=max(1, len(speaker_map)),
        adapter_dim=int(cfg["adapter_dim"]),
        adapter_heads=int(cfg["adapter_heads"]),
        adapter_layers=int(cfg["adapter_layers"]),
        dropout=float(cfg["dropout"]),
        feature_dropout=float(cfg.get("feature_dropout", 0.08)),
        prototype_temperature=float(cfg["prototype_temperature"]),
        prototype_weight=float(effective_cfg.get("prototype_weight", cfg["prototype_weight"])),
        prototype_residual_scale=float(cfg.get("prototype_residual_scale", 0.35)),
        use_prototypes=flags.get("use_prototypes", True),
        use_domain_adversarial=flags.get("use_domain_adversarial", True),
        use_speaker_adversarial=flags.get("use_speaker_adversarial", True),
        grl_lambda=float(cfg["grl_lambda"]),
        global_prototype=flags.get("global_prototype", False),
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    class_counts = np.bincount(encoded.iloc[split["train"]]["label_id"].to_numpy(), minlength=len(cfg["known_labels"]))
    inverse = class_counts.sum() / np.maximum(class_counts, 1)
    class_power = float(cfg.get("class_weight_power", 0.25))
    weights_np = inverse ** class_power
    weights = torch.tensor(weights_np / weights_np.mean(), dtype=torch.float32, device=device)

    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scaler = _grad_scaler(device)
    epochs = int(max_epochs or cfg["epochs"])
    best_state, best_score, stale = None, -float("inf"), 0
    history: list[dict[str, float]] = []
    optimizer_step = 0

    for epoch in range(1, epochs + 1):
        lr = _set_epoch_lr(optimizer, epoch, epochs, cfg)
        progress = (epoch - 1) / max(1, epochs - 1)
        warmup_fraction = min(0.30, float(cfg.get("warmup_epochs", 2)) / max(1, epochs))
        model.set_training_progress(progress, warmup_fraction=warmup_fraction)
        model.train()
        running: list[float] = []
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            labels = batch["label"].to(device)
            corpus_ids = batch["corpus"].to(device)
            speaker_ids = batch["speaker"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = model(x, corpus_ids=corpus_ids, speaker_ids=speaker_ids, labels=labels)
                loss, _ = compute_loss(output, labels, corpus_ids, speaker_ids, effective_cfg, class_weights=weights)

                # One extra forward handles both source-only augmentations.
                same_x, same_y = same_class_cross_corpus_mix(
                    x, labels, corpus_ids,
                    alpha=float(cfg.get("cross_corpus_mixup_alpha", 0.6)),
                    fraction=float(cfg.get("mixup_fraction", 0.5)),
                )
                boundary_x = boundary_mix(
                    x, labels, corpus_ids,
                    alpha=float(cfg.get("boundary_mixup_alpha", 4.0)),
                    fraction=float(cfg.get("mixup_fraction", 0.5)),
                )
                extra_parts = []
                if len(same_x):
                    extra_parts.append(("same", same_x, same_y))
                if len(boundary_x):
                    extra_parts.append(("boundary", boundary_x, None))
                if extra_parts:
                    merged = torch.cat([item[1] for item in extra_parts], dim=0)
                    extra_logits = model(merged)["logits"]
                    cursor = 0
                    for kind, tensor, target in extra_parts:
                        chunk = extra_logits[cursor:cursor + len(tensor)]
                        cursor += len(tensor)
                        if kind == "same" and float(effective_cfg.get("cross_corpus_mixup_weight", 0.0)) > 0:
                            mix_ce = F.cross_entropy(chunk, target, weight=weights, label_smoothing=float(cfg.get("label_smoothing", 0.05)))
                            loss = loss + float(effective_cfg.get("cross_corpus_mixup_weight", 0.18)) * mix_ce
                        elif kind == "boundary" and float(effective_cfg.get("boundary_entropy_weight", 0.0)) > 0:
                            loss = loss + float(effective_cfg.get("boundary_entropy_weight", 0.025)) * uniform_prediction_loss(chunk)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step += 1
            _ema_update(ema_model, model, step=optimizer_step, max_decay=0.995)
            running.append(float(loss.detach().cpu()))

        ema_model.set_training_progress(progress, warmup_fraction=warmup_fraction)
        val_output = collect_outputs(ema_model, val_loader, device)
        domain_f1, pooled_f1 = _validation_scores(val_output)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(running) if running else math.nan),
            "val_domain_macro_f1": domain_f1,
            "val_macro_f1": pooled_f1,
            "lr": lr,
            "prototype_scale": float(model.prototype_scale),
        }
        history.append(row)
        print(
            f"[{holdout}/{ablation_name}/seed={seed}] epoch {epoch:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} domainF1={domain_f1:.4f} pooledF1={pooled_f1:.4f} lr={lr:.2e}"
        )
        if domain_f1 > best_score + 1e-5:
            best_score = domain_f1
            best_state = copy.deepcopy(ema_model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.prototype_scale = 1.0
    model.eval()

    val_output = collect_outputs(model, val_loader, device)
    test_output = collect_outputs(model, test_loader, device)
    calibration = fit_classwise_conformal(
        val_output["logits"],
        val_output["labels"],
        val_output["dispersion"],
        float(cfg["conformal_alpha"]),
        prototype_distance=val_output["prototype_distance"],
        unknown_score_weights=dict(cfg.get("unknown_score_weights", {})),
        known_acceptance=float(cfg.get("unknown_known_acceptance", 0.98)),
        calibration_groups=val_output["corpus"],
    )
    use_conformal = flags.get("use_conformal", True)
    use_shift_guard = flags.get("use_shift_guard", True)
    decisions = predict_with_deferral(
        test_output["logits"],
        test_output["dispersion"],
        calibration,
        prototype_distance=test_output["prototype_distance"],
    )
    if not use_conformal:
        prediction = decisions["prediction"]
        decisions["membership"] = np.eye(len(cfg["known_labels"]), dtype=bool)[prediction]
        decisions["set_sizes"] = np.ones(len(prediction), dtype=int)
    if not use_shift_guard:
        decisions["abstain"] = np.zeros(len(decisions["prediction"]), dtype=bool)
        decisions["decision"] = np.full(len(decisions["prediction"]), "auto_accept", dtype=object)

    labels_np = test_output["labels"].numpy()
    unknown_mask = labels_np < 0
    metric_values = compute_metrics(
        labels_np,
        decisions["probabilities"],
        decisions["prediction"],
        decisions["membership"],
        decisions["dispersion"],
        unknown_mask,
        len(cfg["known_labels"]),
        unknown_score=decisions["unknown_score"],
        abstain=decisions["abstain"],
    )
    metric_values.update({
        "holdout": holdout,
        "ablation": ablation_name,
        "seed": int(seed),
        "best_val_domain_macro_f1": float(best_score),
        "temperature": float(calibration.get("temperature", 1.0)),
        "trainable_parameters": num_trainable_parameters(model),
        "n_source_corpora": len(source_corpora),
        "feature_layers": str(cache_meta.get("feature_layers", [])),
        "implementation_version": "DIMAP-C-v2.0",
        "config_digest": _config_digest(cfg, ablation_name, ablation, holdout, seed, cache_meta),
    })

    run_dir = ensure_dir(output_root / "runs" / f"{ablation_name}__holdout_{holdout}__seed_{seed}")
    torch.save({
        "model": model.state_dict(), "config": cfg, "ablation": ablation, "holdout": holdout,
        "seed": seed, "corpus_map": corpus_map, "speaker_map": speaker_map, "cache_meta": cache_meta,
    }, run_dir / "checkpoint.pt")
    save_json(history, run_dir / "history.json")
    save_json(calibration, run_dir / "calibration.json")
    save_json(metric_values, run_dir / "metrics.json")
    _save_predictions(run_dir / "predictions.csv", test_output, decisions, list(cfg["known_labels"]))
    return {**metric_values, "run_dir": str(run_dir), "history": history, "calibration": calibration, "predictions_path": str(run_dir / "predictions.csv")}


def run_experiments(
    cfg: dict[str, Any],
    manifest: pd.DataFrame,
    feature_cache: FeatureCache,
    device: torch.device,
    output_root: str | Path,
    folds: list[str] | None = None,
    ablation_scope: str = "representative",
    seed: int | None = None,
    max_epochs: int | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    output_root = ensure_dir(output_root)
    base_seed = int(seed if seed is not None else cfg["seed"])
    if seeds is None:
        seeds = [int(s) for s in cfg.get("seeds", [base_seed])]
    if cfg.get("demo"):
        seeds = [base_seed]
    corpora = folds or list(cfg["kaggle_datasets"].keys())
    records: list[dict[str, Any]] = []

    # Full model: repeated seeds for paper-ready uncertainty estimates.
    for holdout in corpora:
        for s in seeds:
            run_seed = int(s)
            existing = _load_completed_run(output_root, cfg, feature_cache, holdout, "full_DIMAP-C", ABLATIONS["full_DIMAP-C"], run_seed)
            records.append(existing if existing is not None else fit_fold(cfg, manifest, feature_cache, holdout, "full_DIMAP-C", ABLATIONS["full_DIMAP-C"], device, output_root, run_seed, max_epochs))

    # Ablations are intentionally run at one fixed seed unless the user asks for
    # full scope, keeping the one-click paper run practical on a laptop GPU.
    if ablation_scope != "none":
        ablations = {k: v for k, v in ABLATIONS.items() if k != "full_DIMAP-C"}
        ablation_holdouts = corpora if ablation_scope == "full" else [corpora[0]]
        for holdout_index, holdout in enumerate(ablation_holdouts):
            for index, (name, flags) in enumerate(ablations.items()):
                run_seed = base_seed + 1000 + holdout_index * 100 + index
                existing = _load_completed_run(output_root, cfg, feature_cache, holdout, name, flags, run_seed)
                records.append(existing if existing is not None else fit_fold(cfg, manifest, feature_cache, holdout, name, flags, device, output_root, run_seed, max_epochs))

    save_json(records, output_root / "all_results.json")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DIMAP-C v2 on cached features.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ablation-scope", choices=["none", "representative", "full"], default="representative")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(Path(cfg["metadata_dir"]) / "manifest.csv")
    stem = str(cfg.get("feature_cache_name", "wavlm_multilayer_segment_stats"))
    feature_cache = FeatureCache(Path(cfg["cache_dir"]) / f"{stem}.npy", Path(cfg["cache_dir"]) / f"{stem}.json")
    run_experiments(cfg, manifest, feature_cache, device_from_arg(args.device), cfg["output_dir"], ablation_scope=args.ablation_scope, seeds=args.seeds)


if __name__ == "__main__":
    main()
