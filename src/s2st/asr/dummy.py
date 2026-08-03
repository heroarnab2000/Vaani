"""Dummy ASR stage.

Returns a deterministic transcript with fake word timings derived from the
audio length. Lets the full pipeline + eval harness run with zero model
downloads. Replace with FasterWhisperASR in Phase 1.
"""
from __future__ import annotations

import numpy as np

from ..interfaces import ASRStage
from ..types import ASRResult, Word


class DummyASR(ASRStage):
    def __init__(self, language: str = "en", words_per_second: float = 2.5):
        self.language = language
        self.words_per_second = words_per_second

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        duration = len(audio) / sample_rate if sample_rate else 0.0
        n_words = max(1, int(duration * self.words_per_second))
        # fabricate evenly spaced words
        step = duration / n_words if n_words else 0.0
        words = [
            Word(
                text=f"word{i}",
                start=i * step,
                end=(i + 1) * step,
                confidence=0.99,
            )
            for i in range(n_words)
        ]
        text = " ".join(w.text for w in words)
        return ASRResult(
            text=text,
            language=self.language,
            words=words,
            audio_duration=duration,
        )
