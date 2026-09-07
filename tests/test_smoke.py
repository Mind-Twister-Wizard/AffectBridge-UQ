from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_manifest import parse_crema, parse_ravdess, parse_savee, parse_tess
from src.calibrate import fit_classwise_conformal, predict_with_deferral
from src.losses import supervised_contrastive_loss
from src.model import AffectBridgeUQ


def test_dimap_forward_shapes() -> None:
    model = AffectBridgeUQ(input_dim=64, num_classes=6, num_corpora=3, num_speakers=8)
    output = model(torch.randn(5, 8, 64), torch.tensor([0, 1, 2, 0, 1]), torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))
    assert output["logits"].shape == (5, 6)
    assert output["dispersion"].shape == (5,)
    assert output["prototype_distance"].shape == (5,)
    assert output["attention"].shape == (5, 8)


def test_contrastive_loss_accepts_half_embeddings() -> None:
    embeddings = torch.randn(8, 16, dtype=torch.float16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(embeddings, labels)
    assert torch.isfinite(loss)


def test_dataset_label_parsers() -> None:
    assert parse_crema("1001_IEO_ANG_HI")[0] == "anger"
    assert parse_ravdess("03-01-08-01-01-01-01")[0] == "surprise"
    assert parse_savee("DC_su01")[0] == "surprise"
    assert parse_tess("OAF_back_ps")[0] == "surprise"


def test_calibration_pipeline_is_finite() -> None:
    torch.manual_seed(0)
    logits = torch.randn(72, 6)
    labels = torch.tensor(np.tile(np.arange(6), 12), dtype=torch.long)
    dispersion = torch.rand(72) * 0.2
    distance = torch.rand(72) * 0.4
    groups = torch.tensor(np.repeat(np.arange(3), 24), dtype=torch.long)
    calibration = fit_classwise_conformal(logits, labels, dispersion, 0.1, distance, calibration_groups=groups)
    decisions = predict_with_deferral(logits, dispersion, calibration, distance)
    assert np.isfinite(decisions["unknown_score"]).all()
    assert decisions["membership"].shape == (72, 6)
    assert 0.19 <= calibration["temperature"] <= 6.01
