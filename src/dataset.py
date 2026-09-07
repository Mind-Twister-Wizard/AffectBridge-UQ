from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedAffectDataset(Dataset):
    def __init__(self, features: np.ndarray, manifest: pd.DataFrame, indices: np.ndarray):
        self.features = features
        self.manifest = manifest.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor | str]:
        row_index = int(self.indices[position])
        row = self.manifest.iloc[row_index]
        feature = np.asarray(self.features[row_index], dtype=np.float32)
        return {
            "x": torch.from_numpy(feature),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "corpus": torch.tensor(int(row["corpus_id"]), dtype=torch.long),
            "speaker": torch.tensor(int(row["speaker_encoded"]), dtype=torch.long),
            "index": torch.tensor(row_index, dtype=torch.long),
            "path": str(row["path"]),
        }
