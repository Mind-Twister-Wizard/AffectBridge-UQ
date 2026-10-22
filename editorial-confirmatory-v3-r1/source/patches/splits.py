"""Editorial confirmatory protocol: isolated source training / selection / calibration.

The held-out target is completely untouched until final evaluation. Speaker-level
splits are preferred, but small or class-sparse source corpora may require an
explicitly recorded stratified utterance fallback (never across target/source).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def _supports_all_classes(frame: pd.DataFrame, idx: np.ndarray, required: set[str]) -> bool:
    return len(idx) > 0 and required.issubset(set(frame.iloc[idx]["label"].astype(str)))


def _source_partition(manifest: pd.DataFrame, indices: np.ndarray, seed: int,
                      selection_fraction: float, calibration_fraction: float):
    subset = manifest.iloc[indices]
    labels = subset["label"].astype(str).to_numpy()
    groups = subset["speaker_id"].astype(str).to_numpy()
    full_classes = set(labels)
    speakers = np.unique(groups)
    frac = float(selection_fraction + calibration_fraction)
    if not (0 < selection_fraction and 0 < calibration_fraction and frac < 1):
        raise ValueError("Invalid split fractions")
    # Prefer speaker disjointness when at least four speakers are present;
    # with four speakers, expect 2/1/1 speakers (train/selection/calibration).
    if len(speakers) >= 4:
        for trial in range(200):
            first = GroupShuffleSplit(n_splits=1, test_size=frac,
                                      random_state=seed + trial * 37)
            train_rel, held_rel = next(first.split(indices, labels, groups))
            held_labels = labels[held_rel]
            held_groups = groups[held_rel]
            if len(np.unique(held_groups)) < 2:
                continue
            second = GroupShuffleSplit(n_splits=1,
                       test_size=calibration_fraction / frac,
                       random_state=seed + 70000 + trial * 41)
            select_in_held, calib_in_held = next(second.split(held_rel, held_labels, held_groups))
            val_rel, cal_rel = held_rel[select_in_held], held_rel[calib_in_held]
            if all(_supports_all_classes(subset, s, full_classes)
                   for s in (train_rel, val_rel, cal_rel)):
                return (indices[train_rel], indices[val_rel], indices[cal_rel],
                        "speaker_group_disjoint")

    # Explicitly documented fallback, mostly expected for two-speaker TESS.
    # Stratification ensures class support in every source-only partition.
    rel = np.arange(len(indices))
    train_rel, rest_rel = train_test_split(rel, test_size=frac,
                                         random_state=seed, stratify=labels)
    val_rel, cal_rel = train_test_split(rest_rel,
                        test_size=calibration_fraction / frac,
                        random_state=seed + 733, stratify=labels[rest_rel])
    return (indices[train_rel], indices[val_rel], indices[cal_rel],
            "source_utterance_stratified_fallback")


def make_lodo_splits(manifest: pd.DataFrame, holdout: str,
                     seed: int = 42, validation_fraction: float = 0.15,
                     calibration_fraction: float = 0.15) -> dict[str, np.ndarray]:
    known = manifest["is_known"].astype(bool).to_numpy()
    corpora = manifest["corpus"].astype(str).to_numpy()
    source_corpora = sorted(set(corpora[known]) - {holdout})
    if not source_corpora or not (corpora == holdout).any():
        raise ValueError("Missing target or source corpus")
    tr, va, ca = [], [], []
    methods = {}
    for offset, corpus in enumerate(source_corpora):
        indices = np.flatnonzero(known & (corpora == corpus))
        if len(indices) < 50:
            raise ValueError(f"Too few known samples in source {corpus}")
        a, b, c, method = _source_partition(manifest, indices,
                            seed + offset * 1009, validation_fraction,
                            calibration_fraction)
        tr.append(a); va.append(b); ca.append(c)
        methods[corpus] = method
    test = np.flatnonzero(corpora == holdout)  # Includes semantic unknowns, test only.
    result = {"train": np.concatenate(tr), "val": np.concatenate(va),
              "cal": np.concatenate(ca), "test": test,
              "methods": methods}
    for name in ("train", "val", "cal"):
        np.random.default_rng(seed + {"train": 1, "val": 2, "cal": 3}[name]).shuffle(result[name])
    keys = ("train", "val", "cal", "test")
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if np.intersect1d(result[a], result[b]).size:
                raise RuntimeError(f"Source/target leakage detected: {a} intersects {b}")
    for corpus in source_corpora:
        for part in ("train", "val", "cal"):
            selected = manifest.iloc[result[part]]
            required = set(manifest.loc[(manifest.corpus == corpus) & (manifest.is_known == 1), "label"])
            subset = selected[selected.corpus == corpus]
            if not required.issubset(set(subset.label)):
                raise RuntimeError(f"{corpus} has missing known labels in {part}")
        if methods[corpus] == "speaker_group_disjoint":
            speaker_sets = [set(manifest.iloc[result[part]].query('corpus == @corpus').speaker_id)
                            for part in ("train", "val", "cal")]
            if any(a & b for i, a in enumerate(speaker_sets) for b in speaker_sets[i+1:]):
                raise RuntimeError(f"Speaker leakage in {corpus}")
    return result


def encode_training_columns(manifest: pd.DataFrame, train_indices: np.ndarray,
                            val_indices: np.ndarray, source_corpora: list[str]):
    corpus_map = {name: i for i, name in enumerate(source_corpora)}
    # Speaker lookup may contain evaluation-only speakers, but identity targets
    # are used only for training examples, never to optimize on validation/cal.
    speakers = sorted(manifest.iloc[np.concatenate([train_indices, val_indices])]
                      ["speaker_id"].astype(str).unique())
    return corpus_map, {name: i for i, name in enumerate(speakers)}


def add_encoded_columns(manifest: pd.DataFrame, corpus_map: dict[str, int],
                       speaker_map: dict[str, int], known_labels: list[str] | None = None):
    result = manifest.copy()
    known_labels = known_labels or ["anger", "disgust", "fear", "happiness", "sadness", "neutral"]
    result["label_id"] = result["label"].map({lab: i for i, lab in enumerate(known_labels)}).fillna(-1).astype(int)
    result["corpus_id"] = result["corpus"].map(corpus_map).fillna(-1).astype(int)
    result["speaker_encoded"] = result["speaker_id"].map(speaker_map).fillna(-1).astype(int)
    return result
