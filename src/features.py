from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .audio import load_audio
from .config import load_config
from .utils import ensure_dir, save_json


class FeatureCache:
    """Memory-mapped compact segment statistics produced by frozen WavLM."""

    def __init__(self, cache_path: str | Path, metadata_path: str | Path):
        self.cache_path = Path(cache_path)
        self.metadata_path = Path(metadata_path)

    def load(self) -> tuple[np.ndarray, dict[str, Any]]:
        return np.load(self.cache_path, mmap_mode="r"), json.loads(self.metadata_path.read_text(encoding="utf-8"))


def _stable_manifest_signature(manifest: pd.DataFrame, cfg: dict[str, Any]) -> str:
    """Content-derived cache key; unlike v1 it does not change just because
    manifest.csv was rewritten with a new filesystem timestamp."""
    columns = [c for c in ["relative_path", "corpus", "label", "speaker_id", "file_size", "file_mtime_ns"] if c in manifest.columns]
    payload = manifest[columns].astype(str).to_csv(index=False).encode("utf-8")
    feature_spec = json.dumps({
        "model": cfg["feature_model"],
        "layers": list(cfg.get("feature_layers", [12])),
        "tokens": int(cfg["segment_tokens"]),
        "sample_rate": int(cfg["sample_rate"]),
        "max_audio_seconds": float(cfg["max_audio_seconds"]),
        "trim_silence": bool(cfg.get("trim_silence", True)),
        "trim_top_db": float(cfg.get("trim_top_db", 35.0)),
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload + b"\n" + feature_spec).hexdigest()


def _segment_statistics(hidden: torch.Tensor, valid_frames: int, tokens: int) -> torch.Tensor:
    hidden = hidden[:max(1, min(valid_frames, hidden.shape[0]))]
    boundaries = torch.linspace(0, hidden.shape[0], tokens + 1, device=hidden.device).long()
    pooled: list[torch.Tensor] = []
    for i in range(tokens):
        start, end = int(boundaries[i]), int(boundaries[i + 1])
        if end <= start:
            end = min(hidden.shape[0], start + 1)
        segment = hidden[start:end]
        pooled.append(torch.cat([segment.mean(dim=0), segment.std(dim=0, unbiased=False)], dim=0))
    return torch.stack(pooled)


@torch.no_grad()
def extract_wavlm_cache(cfg: dict[str, Any], manifest: pd.DataFrame, device: torch.device, force: bool = False) -> FeatureCache:
    cache_dir = ensure_dir(cfg["cache_dir"])
    stem = str(cfg.get("feature_cache_name", "wavlm_multilayer_segment_stats"))
    cache_path = cache_dir / f"{stem}.npy"
    metadata_path = cache_dir / f"{stem}.json"
    manifest_signature = _stable_manifest_signature(manifest, cfg)
    if cache_path.exists() and metadata_path.exists() and not force:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("manifest_signature") == manifest_signature and metadata.get("num_rows") == len(manifest):
                print(f"Using existing feature cache: {cache_path}")
                return FeatureCache(cache_path, metadata_path)
        except Exception:
            pass

    try:
        from transformers import AutoFeatureExtractor, WavLMModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install requirements.txt before extracting WavLM features.") from exc

    model_name = cfg["feature_model"]
    selected_layers = [int(x) for x in cfg.get("feature_layers", [12])]
    print(f"Loading frozen feature encoder: {model_name} | layers={selected_layers}")
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    encoder = WavLMModel.from_pretrained(model_name).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    base_dim = int(encoder.config.hidden_size)
    feature_dim = base_dim * 2 * len(selected_layers)
    tokens = int(cfg["segment_tokens"])
    mmap = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=(len(manifest), tokens, feature_dim))
    batch_size = int(cfg["feature_batch_size"])
    rows = manifest.reset_index(drop=True)
    for start in tqdm(range(0, len(rows), batch_size), desc="Caching multi-layer WavLM statistics"):
        batch = rows.iloc[start:start + batch_size]
        audios: list[np.ndarray] = []
        lengths: list[int] = []
        for path in batch["path"]:
            audio, _ = load_audio(
                path,
                int(cfg["sample_rate"]),
                float(cfg["max_audio_seconds"]),
                bool(cfg.get("trim_silence", True)),
                float(cfg.get("trim_top_db", 35.0)),
            )
            audios.append(audio)
            lengths.append(len(audio))
        inputs = extractor(audios, sampling_rate=int(cfg["sample_rate"]), return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        outputs = encoder(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        for offset, raw_len in enumerate(lengths):
            # Use WavLM's own convolution-length formula when available.
            raw_length_tensor = torch.tensor([raw_len], device=device)
            try:
                valid_frames = int(encoder._get_feat_extract_output_lengths(raw_length_tensor).item())
            except Exception:
                valid_frames = max(1, int(np.ceil(raw_len / 320.0)))
            stats_per_layer = []
            for layer_index in selected_layers:
                if layer_index < 0 or layer_index >= len(hidden_states):
                    raise ValueError(f"feature_layers contains {layer_index}, but WavLM returned {len(hidden_states)} hidden states")
                stats_per_layer.append(_segment_statistics(hidden_states[layer_index][offset].float(), valid_frames, tokens))
            stats = torch.cat(stats_per_layer, dim=-1)
            mmap[start + offset] = stats.cpu().numpy().astype(np.float16)
    mmap.flush()
    metadata = {
        "manifest_signature": manifest_signature,
        "num_rows": len(manifest),
        "tokens": tokens,
        "feature_dim": feature_dim,
        "base_hidden_dim": base_dim,
        "feature_layers": selected_layers,
        "encoder": model_name,
        "dtype": "float16",
        "preprocessing": {"trim_silence": bool(cfg.get("trim_silence", True)), "trim_top_db": float(cfg.get("trim_top_db", 35.0))},
    }
    save_json(metadata, metadata_path)
    return FeatureCache(cache_path, metadata_path)


def create_demo_cache(cfg: dict[str, Any], manifest: pd.DataFrame, seed: int = 42) -> FeatureCache:
    """Create a tiny separable cache for a no-download smoke test."""
    rng = np.random.default_rng(seed)
    cache_dir = ensure_dir(cfg["cache_dir"])
    cache_path = cache_dir / "demo_segment_stats.npy"
    metadata_path = cache_dir / "demo_segment_stats.json"
    tokens, feature_dim = 8, 96
    mmap = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16, shape=(len(manifest), tokens, feature_dim))
    label_to_idx = {label: i for i, label in enumerate(cfg["known_labels"])}
    corpus_to_idx = {name: i for i, name in enumerate(cfg["kaggle_datasets"])}
    for i, row in manifest.reset_index(drop=True).iterrows():
        base = rng.normal(0, 0.25, size=(tokens, feature_dim)).astype(np.float32)
        if row["label"] in label_to_idx:
            j = label_to_idx[row["label"]]
            base[:, j * 6:(j + 1) * 6] += 2.0
        base[:, 48 + corpus_to_idx[row["corpus"]] * 6:54 + corpus_to_idx[row["corpus"]] * 6] += 0.6
        if not row["is_known"]:
            base[:, 80:88] += 2.5
        mmap[i] = base.astype(np.float16)
    mmap.flush()
    metadata = {"manifest_signature": "demo", "num_rows": len(manifest), "tokens": tokens, "feature_dim": feature_dim, "encoder": "synthetic-demo", "dtype": "float16", "feature_layers": [0]}
    save_json(metadata, metadata_path)
    return FeatureCache(cache_path, metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache compact multi-layer WavLM segment statistics.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = pd.read_csv(Path(cfg["metadata_dir"]) / "manifest.csv")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    extract_wavlm_cache(cfg, manifest, device, args.force)


if __name__ == "__main__":
    main()
