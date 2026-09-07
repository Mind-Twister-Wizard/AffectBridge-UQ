from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_lodo_splits(
    manifest: pd.DataFrame,
    holdout: str,
    seed: int = 42,
    validation_fraction: float = 0.15,
) -> dict[str, np.ndarray]:
    """Leave-one-dataset-out split with corpus-balanced source validation.

    v1 used one pooled GroupShuffleSplit, which could make validation dominated by
    the largest corpus. Here each source corpus contributes its own validation
    slice. Speaker grouping is used when a corpus has enough speakers; very
    small-speaker corpora (e.g. TESS) use a stratified utterance fallback so half
    of the corpus is not discarded from training merely to create validation.
    """
    known = manifest["is_known"].astype(bool).to_numpy()
    corpora = manifest["corpus"].astype(str).to_numpy()
    source_corpora = sorted(set(corpora[known]) - {holdout})
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for offset, corpus in enumerate(source_corpora):
        indices = np.flatnonzero(known & (corpora == corpus))
        if len(indices) < 12:
            raise ValueError(f"Not enough source examples for {corpus} in fold {holdout}.")
        subset = manifest.iloc[indices]
        groups = subset["speaker_id"].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 5:
            splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + offset)
            train_rel, val_rel = next(splitter.split(indices, subset["label"], groups))
        else:
            # Preserve all emotion classes in validation for corpora with only a
            # few speakers. This is a source-side tuning split only; the target
            # corpus remains entirely unseen during model selection.
            rel = np.arange(len(indices))
            train_rel, val_rel = train_test_split(
                rel,
                test_size=validation_fraction,
                random_state=seed + offset,
                stratify=subset["label"].to_numpy(),
            )
        train_parts.append(indices[np.asarray(train_rel, dtype=int)])
        val_parts.append(indices[np.asarray(val_rel, dtype=int)])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng = np.random.default_rng(seed)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    test_indices = np.flatnonzero(corpora == holdout)
    return {"train": train_indices, "val": val_indices, "test": test_indices}


def encode_training_columns(manifest: pd.DataFrame, train_indices: np.ndarray, val_indices: np.ndarray, source_corpora: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    corpus_map = {name: i for i, name in enumerate(source_corpora)}
    speakers = sorted(manifest.iloc[np.concatenate([train_indices, val_indices])]["speaker_id"].astype(str).unique())
    speaker_map = {name: i for i, name in enumerate(speakers)}
    return corpus_map, speaker_map


def add_encoded_columns(
    manifest: pd.DataFrame,
    corpus_map: dict[str, int],
    speaker_map: dict[str, int],
    known_labels: list[str] | None = None,
) -> pd.DataFrame:
    result = manifest.copy()
    known_labels = known_labels or ["anger", "disgust", "fear", "happiness", "sadness", "neutral"]
    result["label_id"] = result["label"].map({label: i for i, label in enumerate(known_labels)}).fillna(-1).astype(int)
    result["corpus_id"] = result["corpus"].map(corpus_map).fillna(-1).astype(int)
    result["speaker_encoded"] = result["speaker_id"].map(speaker_map).fillna(-1).astype(int)
    return result
