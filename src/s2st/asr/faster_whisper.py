"""faster-whisper ASR stage (Phase 1).

Wraps CTranslate2 Whisper for transcription with word-level timestamps.
No torch dependency -- CTranslate2 runs Whisper on CPU (INT8) or CUDA, which
is also the start of the optimized/edge path (Phase 4+).

Selected via configs/default.yaml: `stages.asr: faster_whisper`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..audio import resample_linear, to_mono
from ..interfaces import ASRStage
from ..types import ASRResult, Word

TARGET_SR = 16000  # Whisper operates on 16 kHz mono audio


class FasterWhisperASR(ASRStage):
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "en",
        beam_size: int = 5,
    ):
        # lazy import: keeps the dummy path free of heavy deps
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language  # None -> let Whisper auto-detect
        self.beam_size = beam_size
        # distilled models repeat if conditioned on prior text; turn it off for them
        self.condition_on_previous_text = "distil" not in model_name.lower()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        audio16 = resample_linear(to_mono(audio), sample_rate, TARGET_SR)
        duration = len(audio16) / TARGET_SR

        segments, info = self.model.transcribe(
            audio16,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=self.condition_on_previous_text,
        )

        words: List[Word] = []
        texts: List[str] = []
        for seg in segments:  # segments is a generator; iterating runs inference
            if seg.text:
                texts.append(seg.text.strip())
            for w in seg.words or []:
                words.append(
                    Word(
                        text=w.word.strip(),
                        start=float(w.start),
                        end=float(w.end),
                        confidence=float(w.probability),
                    )
                )

        text = " ".join(t for t in texts if t).strip()
        detected = self.language or getattr(info, "language", "") or ""
        return ASRResult(
            text=text,
            language=detected,
            words=words,
            audio_duration=duration,
        )
