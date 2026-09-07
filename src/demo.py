from __future__ import annotations

import pandas as pd


def make_demo_manifest(cfg: dict) -> pd.DataFrame:
    rows = []
    corpora = list(cfg["kaggle_datasets"].keys())
    known_labels = list(cfg["known_labels"])
    unknown_labels = list(cfg["unknown_labels"])
    for corpus in corpora:
        for speaker_num in range(4):
            for label in known_labels:
                for repetition in range(5):
                    rows.append({
                        "path": f"demo://{corpus}/{speaker_num}/{label}/{repetition}.wav",
                        "relative_path": f"{corpus}/{speaker_num}/{label}/{repetition}.wav",
                        "corpus": corpus,
                        "emotion_raw": label,
                        "label": label,
                        "is_known": 1,
                        "is_semantic_unknown": 0,
                        "speaker_id": f"{corpus}:speaker_{speaker_num}",
                        "file_size": 0,
                        "file_mtime_ns": 0,
                    })
        for label in unknown_labels:
            for repetition in range(8):
                rows.append({
                    "path": f"demo://{corpus}/unknown/{label}/{repetition}.wav",
                    "relative_path": f"{corpus}/unknown/{label}/{repetition}.wav",
                    "corpus": corpus,
                    "emotion_raw": label,
                    "label": label,
                    "is_known": 0,
                    "is_semantic_unknown": 1,
                    "speaker_id": f"{corpus}:unknown",
                    "file_size": 0,
                    "file_mtime_ns": 0,
                })
    return pd.DataFrame(rows)
