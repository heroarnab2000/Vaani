"""Small dependency-light audio helpers (mono mix + linear resample).

Linear resampling is intentionally crude -- it is adequate for a baseline
(Whisper and the metrics are robust to it) and keeps the core free of scipy/
librosa. A higher-quality resampler can be swapped in during Phase 4 if it
ever shows up as a quality regression.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if not src_sr or src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * dst_sr / src_sr))
    if n_out <= 0:
        return audio.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_wav(path: Union[str, Path], target_sr: int) -> Optional[np.ndarray]:
    """Load a wav as mono float32 at target_sr, or None if it can't be read."""
    try:
        import soundfile as sf  # optional dep
    except Exception:
        return None
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception:
        return None
    return resample_linear(to_mono(audio), sr, target_sr)


def fit_duration(
    audio: np.ndarray,
    sr: int,
    target_sec: float,
    max_factor: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Time-stretch ``audio`` toward ``target_sec`` (isochrony TTS rate control).

    Uses a phase vocoder (librosa), which changes tempo while preserving pitch,
    so the speaker's voice is kept. The stretch factor is clamped to
    [1/max_factor, max_factor] to avoid artifacts when the text is far longer or
    shorter than the budget -- residual deviation in those clamped cases is what
    duration-aware MT reranking is meant to reduce.

    Returns (stretched_audio, applied_factor) where factor > 1 means the audio
    was sped up (shortened).
    """
    cur = len(audio) / sr if sr else 0.0
    if target_sec <= 0 or cur <= 0:
        return audio.astype(np.float32), 1.0
    factor = cur / target_sec  # >1 -> need to speed up (shorten)
    factor = min(max(factor, 1.0 / max_factor), max_factor)
    if abs(factor - 1.0) < 1e-3:
        return audio.astype(np.float32), 1.0
    import librosa  # heavy; imported only when rate control is active

    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=factor)
    return np.ascontiguousarray(stretched, dtype=np.float32), factor
