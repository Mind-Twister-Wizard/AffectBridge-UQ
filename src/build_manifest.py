from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .utils import ensure_dir

ALIASES = {
    "ang": "anger", "angry": "anger", "anger": "anger",
    "dis": "disgust", "disgust": "disgust", "disgusted": "disgust",
    "fea": "fear", "fear": "fear", "fearful": "fear",
    "hap": "happiness", "happy": "happiness", "happiness": "happiness",
    "sad": "sadness", "sadness": "sadness",
    "neu": "neutral", "neutral": "neutral",
    "cal": "calm", "calm": "calm",
    "sur": "surprise", "su": "surprise", "ps": "surprise",
    "surprised": "surprise", "surprise": "surprise",
    "pleasant_surprise": "surprise", "pleasant surprise": "surprise",
    "bor": "boredom", "boredom": "boredom",
}


def _normalise_token(token: str) -> str | None:
    token = token.lower().strip(" _-.()[]")
    if token in ALIASES:
        return ALIASES[token]
    for alias, label in sorted(ALIASES.items(), key=lambda pair: -len(pair[0])):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", token):
            return label
    return None


def _keyword_label(text: str) -> str | None:
    lowered = text.lower().replace("-", "_")
    for alias, label in sorted(ALIASES.items(), key=lambda pair: -len(pair[0])):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered):
            return label
    return None


def parse_crema(stem: str) -> tuple[str | None, str]:
    parts = stem.split("_")
    raw = parts[2] if len(parts) > 2 else stem
    return _normalise_token(raw) or _keyword_label(stem), parts[0] if parts else "unknown"


def parse_ravdess(stem: str) -> tuple[str | None, str]:
    parts = stem.split("-")
    emotion_ids = {"01": "neutral", "02": "calm", "03": "happiness", "04": "sadness", "05": "anger", "06": "fear", "07": "disgust", "08": "surprise"}
    raw = parts[2] if len(parts) > 2 else ""
    speaker = parts[-1] if parts else "unknown"
    return emotion_ids.get(raw) or _keyword_label(stem), f"actor_{speaker}"


def parse_tess(stem: str) -> tuple[str | None, str]:
    parts = stem.split("_")
    # Standard TESS uses ps for pleasant surprise in some mirrors.
    tail = parts[-1].lower() if parts else ""
    label = _normalise_token(tail) or _keyword_label(stem)
    speaker = parts[0] if parts else "unknown"
    return label, speaker


def parse_savee(stem: str) -> tuple[str | None, str]:
    parts = stem.split("_")
    code_part = parts[1].lower() if len(parts) > 1 else ""
    match = re.match(r"(sa|su|[adfhn])", code_part)
    code = match.group(1) if match else code_part
    code_map = {"a": "anger", "d": "disgust", "f": "fear", "h": "happiness", "n": "neutral", "sa": "sadness", "su": "surprise"}
    return code_map.get(code) or _keyword_label(stem), parts[0] if parts else "unknown"


PARSERS = {"crema": parse_crema, "ravdess": parse_ravdess, "tess": parse_tess, "savee": parse_savee}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    known_labels = list(cfg["known_labels"])
    unknown_labels = list(cfg["unknown_labels"])
    allowed = set(known_labels + unknown_labels)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    raw_root = Path(cfg["raw_dir"])
    for corpus, item in cfg["kaggle_datasets"].items():
        root = raw_root / corpus
        parser = PARSERS[item["parser"]]
        audio_paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"})
        if not audio_paths:
            raise FileNotFoundError(f"No audio files found under {root}. Run with --download first.")
        for path in audio_paths:
            label, speaker = parser(path.stem)
            if label is None:
                # Some Kaggle mirrors encode the emotion in the parent folder
                # rather than the file stem (notably TESS pleasant surprise).
                label = _keyword_label("/".join(path.parts[-3:]))
            if label not in allowed:
                rejected.append({"path": str(path.resolve()), "corpus": corpus, "parsed_label": label or "unparsed", "reason": "label_not_in_protocol"})
                continue
            stat = path.stat()
            rows.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(raw_root)).replace("\\", "/"),
                "corpus": corpus,
                "emotion_raw": label,
                "label": label,
                "is_known": int(label in known_labels),
                "is_semantic_unknown": int(label in unknown_labels),
                "speaker_id": f"{corpus}:{speaker}",
                "file_size": int(stat.st_size),
                "file_mtime_ns": int(stat.st_mtime_ns),
                "content_sha256": _sha256(path),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Manifest is empty after label validation.")
    out_dir = ensure_dir(cfg["metadata_dir"])
    # Reject exact mirrored audio content within each corpus before any split or
    # feature-cache construction. This prevents Kaggle mirrors that contain the
    # same files under multiple directories from silently reweighting training.
    duplicate_mask = frame.duplicated(subset=["corpus", "content_sha256"], keep="first")
    duplicates = frame.loc[duplicate_mask].copy()
    duplicates.to_csv(out_dir / "manifest_duplicates.csv", index=False)
    frame = frame.loc[~duplicate_mask].copy()

    # Sort on stable logical keys so the feature-cache row order is deterministic.
    frame = frame.sort_values(["corpus", "relative_path"]).reset_index(drop=True)
    frame.to_csv(out_dir / "manifest.csv", index=False)
    pd.DataFrame(rejected, columns=["path", "corpus", "parsed_label", "reason"]).to_csv(out_dir / "manifest_rejected.csv", index=False)
    summary = frame.groupby(["corpus", "label"], dropna=False).size().reset_index(name="count")
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    # Fail fast on protocol-breaking parser mistakes rather than silently
    # training on a malformed label map.
    for corpus in cfg["kaggle_datasets"]:
        corpus_rows = frame[frame["corpus"] == corpus]
        missing = [label for label in known_labels if not (corpus_rows["label"] == label).any()]
        if missing:
            raise RuntimeError(f"{corpus}: missing required known labels {missing}. Check dataset mirror/parser before training.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse downloaded corpora into a harmonized manifest.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = build_manifest(cfg)
    print(frame.groupby(["corpus", "label"]).size().to_string())


if __name__ == "__main__":
    main()
