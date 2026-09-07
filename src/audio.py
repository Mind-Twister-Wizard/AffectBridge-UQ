from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    max_seconds: float = 8.0,
    trim_silence: bool = True,
    trim_top_db: float = 35.0,
) -> tuple[np.ndarray, int]:
    """Load mono audio with conservative corpus-normalising preprocessing.

    Silence trimming removes dataset-specific leading/trailing padding while
    intentionally avoiding aggressive loudness normalization, because intensity
    itself carries affective information.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.nan_to_num(audio).astype(np.float32, copy=False)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    if trim_silence and len(audio) >= int(0.25 * sr):
        trimmed, _ = librosa.effects.trim(audio, top_db=float(trim_top_db))
        if len(trimmed) >= int(0.20 * sr):
            audio = trimmed.astype(np.float32, copy=False)
    max_len = int(max_seconds * target_sr)
    if len(audio) > max_len:
        start = (len(audio) - max_len) // 2
        audio = audio[start:start + max_len]
    # Protect against corrupt / clipped waveforms, but do not normalize normal
    # utterances to unit peak (that would erase useful intensity differences).
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    if len(audio) < int(0.25 * target_sr):
        audio = np.pad(audio, (0, int(0.25 * target_sr) - len(audio)))
    return audio.astype(np.float32, copy=False), sr
