"""Dummy TTS stage.

Generates a low-amplitude sine tone whose length equals the requested
duration budget. This lets us measure the isochrony metric (realized vs
target duration) end-to-end even before a real vocoder exists. Replace with
XTTSv2TTS in Phase 1.
"""
from __future__ import annotations

import numpy as np

from ..interfaces import TTSStage
from ..types import TranslationResult, TTSResult


class DummyTTS(TTSStage):
    def __init__(self, sample_rate: int = 16000, chars_per_second: float = 14.0):
        self.sample_rate = sample_rate
        self.chars_per_second = chars_per_second

    def synthesize(
        self,
        translation: TranslationResult,
        speaker_wav: np.ndarray | None,
        sample_rate: int,
    ) -> TTSResult:
        sr = sample_rate or self.sample_rate
        # naive duration estimate from text length; deliberately NOT matched
        # to the budget, so Phase 2's rate control has something to fix.
        est_duration = max(0.1, len(translation.text) / self.chars_per_second)
        n = int(est_duration * sr)
        t = np.linspace(0, est_duration, n, endpoint=False)
        audio = (0.01 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        return TTSResult(audio=audio, sample_rate=sr, realized_duration=est_duration)
