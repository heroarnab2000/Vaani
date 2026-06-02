"""Abstract stage interfaces.

Every real model (Whisper, NLLB, XTTS, ...) implements one of these. The
orchestrator and eval harness only ever talk to these interfaces, so you can
swap a dummy for faster-whisper for TensorRT without touching anything else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import ASRResult, TranslationResult, TTSResult


class ASRStage(ABC):
    name: str = "asr"

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        ...


class TranslationStage(ABC):
    name: str = "translation"

    @abstractmethod
    def translate(
        self,
        asr: ASRResult,
        tgt_lang: str,
        target_speech_duration: float,
    ) -> TranslationResult:
        """Translate, aiming to fit `target_speech_duration` (isochrony).

        Dummy implementation ignores the duration; real implementations in
        Phase 2 use it via length-control tokens or candidate reranking.
        """
        ...


class TTSStage(ABC):
    name: str = "tts"

    @abstractmethod
    def synthesize(
        self,
        translation: TranslationResult,
        speaker_wav: np.ndarray | None,
        sample_rate: int,
    ) -> TTSResult:
        ...
