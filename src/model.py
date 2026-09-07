from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, coefficient: float = 0.2):
        super().__init__()
        self.coefficient = float(coefficient)

    def set_coefficient(self, coefficient: float) -> None:
        self.coefficient = float(coefficient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.coefficient)


class TemporalAdapter(nn.Module):
    """Compact dual-scale temporal adapter over cached SSL statistics."""

    def __init__(
        self,
        input_dim: int,
        dim: int = 192,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.20,
        feature_dropout: float = 0.08,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.projection = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Dropout(dropout * 0.5))
        self.short_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.long_conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.attention_pool = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_dropout(self.input_norm(x))
        x = self.projection(x)
        y = x.transpose(1, 2)
        short = self.short_conv(y).transpose(1, 2)
        long = self.long_conv(y).transpose(1, 2)
        gate = self.gate(torch.cat([short, long], dim=-1))
        x = x + gate * short + (1.0 - gate) * long
        x = self.transformer(x)
        weights = torch.softmax(self.attention_pool(x).squeeze(-1), dim=-1)
        pooled = self.norm(torch.sum(x * weights.unsqueeze(-1), dim=1))
        return pooled, weights


class AffectBridgeUQ(nn.Module):
    """DIMAP-C v2: domain-invariant hierarchical multi-prototype network.

    The key change from the exploratory implementation is a global emotion
    prototype anchored by small corpus-specific residual prototypes. This keeps
    the multi-prototype idea while explicitly discouraging each corpus from
    learning an unrelated emotion geometry.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_corpora: int,
        num_speakers: int,
        adapter_dim: int = 192,
        adapter_heads: int = 4,
        adapter_layers: int = 2,
        dropout: float = 0.20,
        feature_dropout: float = 0.08,
        prototype_temperature: float = 0.16,
        prototype_weight: float = 0.16,
        prototype_residual_scale: float = 0.35,
        use_prototypes: bool = True,
        use_domain_adversarial: bool = True,
        use_speaker_adversarial: bool = True,
        grl_lambda: float = 0.15,
        global_prototype: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_corpora = int(max(1, num_corpora))
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_weight = float(prototype_weight)
        self.prototype_residual_scale = float(prototype_residual_scale)
        self.use_prototypes = bool(use_prototypes)
        self.use_domain_adversarial = bool(use_domain_adversarial)
        self.use_speaker_adversarial = bool(use_speaker_adversarial)
        self.global_prototype = bool(global_prototype)
        self.prototype_scale = 1.0
        self.max_grl_lambda = float(grl_lambda)

        self.adapter = TemporalAdapter(input_dim, adapter_dim, adapter_heads, adapter_layers, dropout, feature_dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(adapter_dim), nn.Dropout(dropout), nn.Linear(adapter_dim, num_classes))

        self.global_prototypes = nn.Parameter(torch.randn(num_classes, adapter_dim) * 0.035)
        self.corpus_residuals = nn.Parameter(torch.randn(self.num_corpora, num_classes, adapter_dim) * 0.012)
        self.prototype_gate = nn.Sequential(nn.LayerNorm(adapter_dim), nn.Linear(adapter_dim, 1), nn.Sigmoid())

        self.corpus_grl = GradientReversal(0.0)
        self.speaker_grl = GradientReversal(0.0)
        self.corpus_head = nn.Sequential(self.corpus_grl, nn.LayerNorm(adapter_dim), nn.Linear(adapter_dim, self.num_corpora))
        self.speaker_head = nn.Sequential(self.speaker_grl, nn.LayerNorm(adapter_dim), nn.Linear(adapter_dim, max(1, num_speakers)))

    def set_training_progress(self, progress: float, warmup_fraction: float = 0.08) -> None:
        """Progressively enable prototypes and adversarial invariance.

        Random prototypes and a full-strength GRL at epoch 1 made v1 unstable.
        A smooth schedule lets the emotion classifier establish a useful space
        before invariance constraints become strong.
        """
        p = float(max(0.0, min(1.0, progress)))
        if warmup_fraction > 0:
            self.prototype_scale = float(max(0.0, min(1.0, (p - warmup_fraction) / max(1e-6, 1.0 - warmup_fraction))))
        else:
            self.prototype_scale = p
        grl = self.max_grl_lambda * (2.0 / (1.0 + math.exp(-8.0 * p)) - 1.0)
        self.corpus_grl.set_coefficient(grl)
        self.speaker_grl.set_coefficient(grl)

    def prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        global_proto = self.global_prototypes
        if self.global_prototype:
            local_proto = global_proto.unsqueeze(0).expand(self.num_corpora, -1, -1)
        else:
            local_proto = global_proto.unsqueeze(0) + self.prototype_residual_scale * torch.tanh(self.corpus_residuals)
        return global_proto, local_proto

    def forward(
        self,
        x: torch.Tensor,
        corpus_ids: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding, attention = self.adapter(x)
        classifier_logits = self.classifier(embedding)
        global_proto, local_proto = self.prototypes()

        if self.use_prototypes:
            z = F.normalize(embedding, dim=-1)
            pg = F.normalize(global_proto, dim=-1)
            pl = F.normalize(local_proto, dim=-1)
            global_sim = torch.einsum("bd,cd->bc", z, pg)
            local_sim = torch.einsum("bd,ncd->bnc", z, pl)
            local_logits = torch.logsumexp(local_sim / self.prototype_temperature, dim=1) - math.log(local_sim.shape[1])
            global_logits = global_sim / self.prototype_temperature
            proto_logits = 0.55 * global_logits + 0.45 * local_logits
            proto_gate = 0.25 + 0.75 * self.prototype_gate(embedding).squeeze(-1)
            logits = classifier_logits + self.prototype_scale * self.prototype_weight * proto_gate.unsqueeze(-1) * proto_logits
            dispersion_by_class = local_sim.std(dim=1, unbiased=False) if local_sim.shape[1] > 1 else torch.zeros_like(local_sim[:, 0])
            risk_index = logits.argmax(dim=-1)
            dispersion = dispersion_by_class.gather(1, risk_index.unsqueeze(1)).squeeze(1)
            prototype_distance = 1.0 - global_sim.max(dim=1).values
        else:
            global_sim = embedding.new_zeros((embedding.shape[0], self.num_classes))
            local_sim = embedding.new_zeros((embedding.shape[0], self.num_corpora, self.num_classes))
            proto_logits = embedding.new_zeros((embedding.shape[0], self.num_classes))
            proto_gate = embedding.new_zeros(embedding.shape[0])
            dispersion_by_class = embedding.new_zeros((embedding.shape[0], self.num_classes))
            dispersion = embedding.new_zeros(embedding.shape[0])
            prototype_distance = embedding.new_ones(embedding.shape[0])
            logits = classifier_logits

        source_proto = embedding.new_zeros(embedding.shape)
        global_source_proto = embedding.new_zeros(embedding.shape)
        if corpus_ids is not None and labels is not None and self.use_prototypes:
            safe_ids = corpus_ids.clamp(0, self.num_corpora - 1)
            safe_labels = labels.clamp(0, self.num_classes - 1)
            source_proto = local_proto[safe_ids, safe_labels]
            global_source_proto = global_proto[safe_labels]

        output = {
            "embedding": embedding,
            "attention": attention,
            "classifier_logits": classifier_logits,
            "logits": logits,
            "proto_logits": proto_logits,
            "global_proto_sim": global_sim,
            "proto_sim": local_sim,
            "dispersion_by_class": dispersion_by_class,
            "dispersion": dispersion,
            "prototype_distance": prototype_distance,
            "prototype_gate": proto_gate,
            "source_proto": source_proto,
            "global_source_proto": global_source_proto,
            "global_prototypes": global_proto,
            "local_prototypes": local_proto,
        }
        if self.use_domain_adversarial:
            output["corpus_logits"] = self.corpus_head(embedding)
        if self.use_speaker_adversarial:
            output["speaker_logits"] = self.speaker_head(embedding)
        return output
