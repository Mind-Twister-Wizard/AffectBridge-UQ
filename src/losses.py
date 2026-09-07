from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    # Keep pairwise similarity in float32 under CUDA autocast.
    z = F.normalize(embeddings.float(), dim=-1)
    similarity = torch.matmul(z, z.T) / temperature
    mask_self = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~mask_self
    similarity = similarity.masked_fill(mask_self, torch.finfo(similarity.dtype).min)
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.new_zeros(())
    mean_log_prob_pos = (log_prob * positive).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def conditional_centroid_alignment_loss(embeddings: torch.Tensor, labels: torch.Tensor, corpus_ids: torch.Tensor) -> torch.Tensor:
    """Align same-emotion centroids across source corpora.

    This is a stable class-conditional first-moment alignment that avoids the
    negative transfer of forcing unlike emotions from different corpora to match.
    """
    z = F.normalize(embeddings.float(), dim=-1)
    losses: list[torch.Tensor] = []
    for label in labels.unique():
        label_mask = labels == label
        corpora = corpus_ids[label_mask].unique()
        centroids: list[torch.Tensor] = []
        for corpus in corpora:
            mask = label_mask & (corpus_ids == corpus)
            if int(mask.sum()) >= 2:
                centroids.append(F.normalize(z[mask].mean(dim=0, keepdim=True), dim=-1).squeeze(0))
        if len(centroids) >= 2:
            stack = torch.stack(centroids)
            anchor = F.normalize(stack.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            losses.append((1.0 - (stack * anchor).sum(dim=-1)).mean())
    if not losses:
        return embeddings.new_zeros(())
    return torch.stack(losses).mean().to(embeddings.dtype)


def prototype_consistency_loss(global_prototypes: torch.Tensor, local_prototypes: torch.Tensor) -> torch.Tensor:
    g = F.normalize(global_prototypes.float(), dim=-1).unsqueeze(0)
    l = F.normalize(local_prototypes.float(), dim=-1)
    return (1.0 - (g * l).sum(dim=-1)).mean().to(global_prototypes.dtype)


def uniform_prediction_loss(logits: torch.Tensor) -> torch.Tensor:
    """KL(p || Uniform), minimized for deliberately ambiguous pseudo-OOD mixes."""
    probs = torch.softmax(logits.float(), dim=-1).clamp_min(1e-8)
    return (probs * torch.log(probs * probs.shape[-1])).sum(dim=-1).mean().to(logits.dtype)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    corpus_ids: torch.Tensor,
    speaker_ids: torch.Tensor,
    cfg: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(
        outputs["logits"],
        labels,
        weight=class_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    total = ce
    parts = {"classification": float(ce.detach().cpu())}

    if cfg.get("use_prototypes", True):
        proto_ce = F.cross_entropy(outputs["proto_logits"], labels, weight=class_weights, label_smoothing=0.02)
        global_align = 1.0 - F.cosine_similarity(outputs["embedding"], outputs["global_source_proto"], dim=-1).mean()
        local_align = 1.0 - F.cosine_similarity(outputs["embedding"], outputs["source_proto"], dim=-1).mean()
        align = 0.65 * global_align + 0.35 * local_align
        consistency = prototype_consistency_loss(outputs["global_prototypes"], outputs["local_prototypes"])
        contrastive = supervised_contrastive_loss(outputs["embedding"], labels)
        total = total + float(cfg.get("prototype_weight", 0.16)) * 0.35 * proto_ce
        total = total + float(cfg.get("alignment_weight", 0.08)) * align
        total = total + float(cfg.get("prototype_consistency_weight", 0.04)) * consistency
        total = total + float(cfg.get("contrastive_weight", 0.08)) * contrastive
        parts.update({
            "prototype_ce": float(proto_ce.detach().cpu()),
            "alignment": float(align.detach().cpu()),
            "prototype_consistency": float(consistency.detach().cpu()),
            "contrastive": float(contrastive.detach().cpu()),
        })

    conditional = conditional_centroid_alignment_loss(outputs["embedding"], labels, corpus_ids)
    total = total + float(cfg.get("conditional_alignment_weight", 0.05)) * conditional
    parts["conditional_alignment"] = float(conditional.detach().cpu())

    if cfg.get("use_domain_adversarial", True) and "corpus_logits" in outputs:
        domain_ce = F.cross_entropy(outputs["corpus_logits"], corpus_ids)
        total = total + float(cfg.get("domain_adversarial_weight", cfg.get("adversarial_weight", 0.04))) * domain_ce
        parts["corpus_adversarial"] = float(domain_ce.detach().cpu())
    if cfg.get("use_speaker_adversarial", True) and "speaker_logits" in outputs:
        speaker_ce = F.cross_entropy(outputs["speaker_logits"], speaker_ids)
        total = total + float(cfg.get("speaker_adversarial_weight", 0.015)) * speaker_ce
        parts["speaker_adversarial"] = float(speaker_ce.detach().cpu())
    parts["total"] = float(total.detach().cpu())
    return total, parts
