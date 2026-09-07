from __future__ import annotations

from typing import Any

import numpy as np
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one scalar temperature on source validation NLL only."""
    if len(labels) == 0:
        return 1.0
    logits_np = logits.detach().float().cpu().numpy()
    y = labels.detach().cpu().numpy().astype(int)
    keep = y >= 0
    if not keep.any():
        return 1.0
    logits_np = logits_np[keep]
    y = y[keep]

    def nll(temp: float) -> float:
        t = max(0.05, float(temp))
        z = logits_np / t
        z = z - z.max(axis=1, keepdims=True)
        logsum = np.log(np.exp(z).sum(axis=1) + 1e-12)
        return float(np.mean(logsum - z[np.arange(len(y)), y]))

    # A deterministic dense log-grid is robust and avoids optimizer failures.
    grid = np.geomspace(0.20, 6.0, 121)
    losses = np.asarray([nll(t) for t in grid])
    best = int(np.argmin(losses))
    return float(grid[best])


def _softmax_np(logits: torch.Tensor, temperature: float) -> np.ndarray:
    return torch.softmax(logits.float() / float(max(0.05, temperature)), dim=-1).detach().cpu().numpy()


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) * 1.4826)
    if mad < 1e-6:
        mad = float(np.std(values))
    return median, max(mad, 1e-6)


def _risk_components(probabilities: np.ndarray, prototype_distance: np.ndarray, dispersion: np.ndarray) -> dict[str, np.ndarray]:
    max_p = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1) / np.log(probabilities.shape[1])
    return {
        "confidence": 1.0 - max_p,
        "entropy": entropy,
        "prototype_distance": np.asarray(prototype_distance, dtype=float),
        "dispersion": np.asarray(dispersion, dtype=float),
    }


def _score_from_reference(components: dict[str, np.ndarray], reference: dict[str, Any], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros(len(next(iter(components.values()))), dtype=float)
    for name, values in components.items():
        loc = float(reference[name]["location"])
        scale = float(reference[name]["scale"])
        score += float(weights.get(name, 0.0)) * ((values - loc) / max(scale, 1e-6))
    return score


def fit_classwise_conformal(
    logits: torch.Tensor,
    labels: torch.Tensor,
    dispersion: torch.Tensor,
    alpha: float = 0.1,
    prototype_distance: torch.Tensor | None = None,
    unknown_score_weights: dict[str, float] | None = None,
    known_acceptance: float = 0.98,
    calibration_groups: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Temperature-scaled classwise conformal calibration + source-only risk guard."""
    temperature = fit_temperature(logits, labels)
    probs = _softmax_np(logits, temperature)
    y = labels.detach().cpu().numpy()
    d = dispersion.detach().float().cpu().numpy()
    pdist = prototype_distance.detach().float().cpu().numpy() if prototype_distance is not None else np.zeros_like(d)
    n_classes = probs.shape[1]
    quantiles: list[float] = []
    for c in range(n_classes):
        scores = 1.0 - probs[y == c, c]
        if len(scores) == 0:
            valid = y >= 0
            scores = 1.0 - probs[np.arange(len(y))[valid], y[valid]]
        rank = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / max(1, len(scores)))
        try:
            q = float(np.quantile(scores, rank, method="higher"))
        except TypeError:  # numpy<1.22 compatibility
            q = float(np.quantile(scores, rank, interpolation="higher"))
        quantiles.append(float(np.clip(q, 0.0, 1.0)))

    components = _risk_components(probs, pdist, d)
    reference: dict[str, Any] = {}
    for name, values in components.items():
        location, scale = _robust_location_scale(values)
        reference[name] = {"location": location, "scale": scale}
    weights = unknown_score_weights or {"confidence": 1.0, "entropy": 0.5, "prototype_distance": 0.8, "dispersion": 0.2}
    val_risk = _score_from_reference(components, reference, weights)
    if len(val_risk):
        if calibration_groups is not None:
            groups = calibration_groups.detach().cpu().numpy()
            group_thresholds = [float(np.quantile(val_risk[groups == g], float(known_acceptance))) for g in np.unique(groups) if np.any(groups == g)]
            threshold = max(group_thresholds) if group_thresholds else float(np.quantile(val_risk, float(known_acceptance)))
        else:
            group_thresholds = []
            threshold = float(np.quantile(val_risk, float(known_acceptance)))
    else:
        group_thresholds = []
        threshold = float("inf")
    return {
        "alpha": float(alpha),
        "temperature": float(temperature),
        "class_nonconformity_quantiles": quantiles,
        "risk_reference": reference,
        "risk_weights": {k: float(v) for k, v in weights.items()},
        "risk_threshold": threshold,
        "risk_group_thresholds": group_thresholds,
        "known_acceptance": float(known_acceptance),
    }


def predict_with_deferral(
    logits: torch.Tensor,
    dispersion: torch.Tensor,
    calibration: dict[str, Any],
    prototype_distance: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    temperature = float(calibration.get("temperature", 1.0))
    probs = _softmax_np(logits, temperature)
    d = dispersion.detach().float().cpu().numpy()
    pdist = prototype_distance.detach().float().cpu().numpy() if prototype_distance is not None else np.zeros_like(d)
    q = np.asarray(calibration["class_nonconformity_quantiles"], dtype=float)
    membership = probs >= (1.0 - q)[None, :]
    set_sizes = membership.sum(axis=1).astype(int)
    max_probability = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    components = _risk_components(probs, pdist, d)
    unknown_score = _score_from_reference(components, calibration["risk_reference"], calibration["risk_weights"])
    high_shift_risk = unknown_score > float(calibration.get("risk_threshold", np.inf))
    empty = set_sizes == 0
    ambiguous = set_sizes > 1
    abstain = empty | ambiguous | high_shift_risk
    decision = np.where(empty, "unknown", np.where(ambiguous | high_shift_risk, "human_review", "auto_accept"))
    return {
        "probabilities": probs,
        "prediction": pred,
        "membership": membership,
        "set_sizes": set_sizes,
        "max_probability": max_probability,
        "dispersion": d,
        "prototype_distance": pdist,
        "unknown_score": unknown_score,
        "high_shift_risk": high_shift_risk,
        "abstain": abstain,
        "decision": decision,
        "temperature": np.full(len(pred), temperature, dtype=float),
    }
