"""Core data types passed between pipeline stages.

Keeping these in one place means every stage agrees on the contract, and the
eval harness can introspect timing/quality fields uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Word:
    """A single recognized word with timing, used for duration control + lip-sync."""
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 1.0


@dataclass
class ASRResult:
    text: str
    language: str
    words: List[Word] = field(default_factory=list)
    audio_duration: float = 0.0  # seconds of source speech

    @property
    def speech_duration(self) -> float:
        """Duration of actual speech (first word start -> last word end).

        This is the quantity isochrony control tries to match in the target
        language, not the raw clip length.
        """
        if not self.words:
            return self.audio_duration
        return self.words[-1].end - self.words[0].start


@dataclass
class TranslationResult:
    text: str
    src_lang: str
    tgt_lang: str
    target_speech_duration: float  # the duration budget handed to TTS
    # diagnostics for the isochrony ablation
    n_candidates: int = 1
    chosen_length_bucket: Optional[str] = None


@dataclass
class TTSResult:
    audio: np.ndarray          # waveform, float32 mono
    sample_rate: int
    realized_duration: float   # seconds actually produced


@dataclass
class S2STOutput:
    """Everything the pipeline produced for one input, plus per-stage timings."""
    asr: ASRResult
    translation: TranslationResult
    tts: TTSResult
    # wall-clock seconds per stage, filled in by the orchestrator
    timings: dict = field(default_factory=dict)

    @property
    def total_latency(self) -> float:
        return sum(self.timings.values())
