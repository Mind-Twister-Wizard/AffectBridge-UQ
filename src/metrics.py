from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += mask.mean() * abs((predictions[mask] == labels[mask]).mean() - confidence[mask].mean())
    return float(ece)


def unknown_fpr95(scores: np.ndarray, unknown_mask: np.ndarray) -> float:
    known_scores = scores[~unknown_mask]
    unknown_scores = scores[unknown_mask]
    if len(known_scores) == 0 or len(unknown_scores) == 0:
        return float("nan")
    threshold = float(np.quantile(unknown_scores, 0.05))
    return float((known_scores >= threshold).mean())


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    prediction: np.ndarray,
    membership: np.ndarray,
    dispersion: np.ndarray,
    unknown_mask: np.ndarray,
    known_classes: int = 6,
    unknown_score: np.ndarray | None = None,
    abstain: np.ndarray | None = None,
) -> dict[str, Any]:
    known_mask = ~unknown_mask
    metrics: dict[str, Any] = {
        "n_total": int(len(labels)),
        "n_known": int(known_mask.sum()),
        "n_unknown": int(unknown_mask.sum()),
    }
    if known_mask.any():
        yk, pk = labels[known_mask], probabilities[known_mask]
        predk = prediction[known_mask]
        set_sizes = membership[known_mask].sum(axis=1)
        metrics.update({
            "accuracy": float(accuracy_score(yk, predk)),
            "macro_f1": float(f1_score(yk, predk, average="macro", zero_division=0)),
            "uar": float(balanced_accuracy_score(yk, predk)),
            "balanced_accuracy": float(balanced_accuracy_score(yk, predk)),
            "nll": float(log_loss(yk, pk, labels=list(range(known_classes)))),
            "brier": float(np.mean(np.sum((pk - np.eye(known_classes)[yk]) ** 2, axis=1))),
            "ece": expected_calibration_error(pk, yk),
            "coverage": float(np.mean(membership[known_mask, yk])),
            "avg_prediction_set_size": float(set_sizes.mean()),
            "singleton_rate": float((set_sizes == 1).mean()),
        })
        if abstain is None:
            accepted = set_sizes == 1
        else:
            accepted = ~np.asarray(abstain, dtype=bool)[known_mask]
        metrics["auto_coverage"] = float(accepted.mean())
        metrics["selective_risk"] = float(1.0 - accuracy_score(yk[accepted], predk[accepted])) if accepted.any() else float("nan")
        metrics["selective_macro_f1"] = float(f1_score(yk[accepted], predk[accepted], average="macro", zero_division=0)) if accepted.any() else float("nan")
        metrics["confusion_matrix"] = confusion_matrix(yk, predk, labels=list(range(known_classes))).tolist()
    else:
        for key in (
            "accuracy", "macro_f1", "uar", "balanced_accuracy", "nll", "brier", "ece", "coverage",
            "avg_prediction_set_size", "singleton_rate", "auto_coverage", "selective_risk", "selective_macro_f1",
        ):
            metrics[key] = float("nan")
        metrics["confusion_matrix"] = np.zeros((known_classes, known_classes), dtype=int).tolist()

    if unknown_mask.any() and (~unknown_mask).any():
        if unknown_score is None:
            max_disp = float(np.max(dispersion)) if len(dispersion) else 0.0
            unknown_score = 1.0 - probabilities.max(axis=1) + dispersion / (max_disp + 1e-8)
        unknown_score = np.asarray(unknown_score, dtype=float)
        binary_true = unknown_mask.astype(int)
        metrics["unknown_auroc"] = float(roc_auc_score(binary_true, unknown_score))
        metrics["unknown_aupr"] = float(average_precision_score(binary_true, unknown_score))
        metrics["unknown_fpr95"] = unknown_fpr95(unknown_score, unknown_mask)
    else:
        metrics["unknown_auroc"] = float("nan")
        metrics["unknown_aupr"] = float("nan")
        metrics["unknown_fpr95"] = float("nan")
    return metrics
