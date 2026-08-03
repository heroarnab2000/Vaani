"""Dummy translation stage.

Echoes the source text with a tag. Crucially, it already threads the
`target_speech_duration` budget through and records diagnostics, so the
isochrony ablation in Phase 2 only needs to make the budget actually
influence generation, not rewire the pipeline.
"""
from __future__ import annotations

from ..interfaces import TranslationStage
from ..types import ASRResult, TranslationResult


class DummyTranslation(TranslationStage):
    def __init__(self, default_tgt: str = "hi"):
        self.default_tgt = default_tgt

    def translate(
        self,
        asr: ASRResult,
        tgt_lang: str,
        target_speech_duration: float,
    ) -> TranslationResult:
        # placeholder "translation": just mark it
        text = f"[{tgt_lang}] " + asr.text
        return TranslationResult(
            text=text,
            src_lang=asr.language,
            tgt_lang=tgt_lang,
            target_speech_duration=target_speech_duration,
            n_candidates=1,
            chosen_length_bucket="normal",
        )
