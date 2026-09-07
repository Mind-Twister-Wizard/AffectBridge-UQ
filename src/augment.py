from __future__ import annotations

import numpy as np
import torch


def _sample_beta(alpha: float, n: int, device: torch.device) -> torch.Tensor:
    if n <= 0:
        return torch.empty(0, device=device)
    distribution = torch.distributions.Beta(float(alpha), float(alpha))
    return distribution.sample((n,)).to(device)


def same_class_cross_corpus_mix(
    x: torch.Tensor,
    labels: torch.Tensor,
    corpora: torch.Tensor,
    alpha: float = 0.6,
    fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create source-only same-emotion mixes between different corpora."""
    candidates: list[tuple[int, int]] = []
    for i in range(len(labels)):
        mask = (labels == labels[i]) & (corpora != corpora[i])
        js = torch.nonzero(mask, as_tuple=False).flatten()
        if len(js):
            j = int(js[torch.randint(len(js), (1,), device=js.device)].item())
            candidates.append((i, j))
    if not candidates:
        return x[:0], labels[:0]
    n = max(1, int(round(len(candidates) * float(fraction))))
    order = torch.randperm(len(candidates), device=x.device)[:n].tolist()
    pairs = [candidates[k] for k in order]
    first = torch.tensor([p[0] for p in pairs], device=x.device, dtype=torch.long)
    second = torch.tensor([p[1] for p in pairs], device=x.device, dtype=torch.long)
    lam = _sample_beta(alpha, len(pairs), x.device)
    # Avoid near-copies; domain interpolation is strongest away from 0/1.
    lam = 0.2 + 0.6 * lam
    shape = [len(pairs)] + [1] * (x.ndim - 1)
    mixed = lam.view(*shape) * x[first] + (1.0 - lam).view(*shape) * x[second]
    return mixed, labels[first]


def boundary_mix(
    x: torch.Tensor,
    labels: torch.Tensor,
    corpora: torch.Tensor,
    alpha: float = 4.0,
    fraction: float = 0.5,
) -> torch.Tensor:
    """Different-emotion cross-corpus mixes used as source-only pseudo-OOD.

    No calm/surprise utterance is used. The resulting samples lie near semantic
    class boundaries and are trained toward high-entropy predictions.
    """
    candidates: list[tuple[int, int]] = []
    for i in range(len(labels)):
        mask = (labels != labels[i]) & (corpora != corpora[i])
        js = torch.nonzero(mask, as_tuple=False).flatten()
        if len(js):
            j = int(js[torch.randint(len(js), (1,), device=js.device)].item())
            candidates.append((i, j))
    if not candidates:
        return x[:0]
    n = max(1, int(round(len(candidates) * float(fraction))))
    order = torch.randperm(len(candidates), device=x.device)[:n].tolist()
    pairs = [candidates[k] for k in order]
    first = torch.tensor([p[0] for p in pairs], device=x.device, dtype=torch.long)
    second = torch.tensor([p[1] for p in pairs], device=x.device, dtype=torch.long)
    lam = _sample_beta(alpha, len(pairs), x.device)
    # Beta(4,4) already concentrates near 0.5; keep it inside [0.3, 0.7].
    lam = 0.3 + 0.4 * lam
    shape = [len(pairs)] + [1] * (x.ndim - 1)
    return lam.view(*shape) * x[first] + (1.0 - lam).view(*shape) * x[second]
