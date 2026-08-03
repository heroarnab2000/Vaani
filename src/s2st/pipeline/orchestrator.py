"""Pipeline orchestrator.

Runs ASR -> (duration estimate) -> Translation -> TTS, timing each stage.
The orchestrator is model-agnostic: it only knows the abstract interfaces,
so the same code path serves dummy, baseline, and optimized stages.
"""
from __future__ import annotations

import time

import numpy as np

from ..interfaces import ASRStage, TranslationStage, TTSStage
from ..types import S2STOutput


class S2STPipeline:
    def __init__(self, asr: ASRStage, translation: TranslationStage, tts: TTSStage):
        self.asr = asr
        self.translation = translation
        self.tts = tts

    def run(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tgt_lang: str = "hi",
        speaker_wav: np.ndarray | None = None,
    ) -> S2STOutput:
        timings: dict = {}

        t0 = time.perf_counter()
        asr_res = self.asr.transcribe(audio, sample_rate)
        timings["asr"] = time.perf_counter() - t0

        # duration budget for isochrony: target speech should ~match source speech
        target_dur = asr_res.speech_duration

        t0 = time.perf_counter()
        tr_res = self.translation.translate(asr_res, tgt_lang, target_dur)
        timings["translation"] = time.perf_counter() - t0

        # voice preservation: clone the source speaker unless a separate
        # reference is given, so the output keeps the input speaker's voice.
        ref_wav = speaker_wav if speaker_wav is not None else audio

        t0 = time.perf_counter()
        tts_res = self.tts.synthesize(tr_res, ref_wav, sample_rate)
        timings["tts"] = time.perf_counter() - t0

        return S2STOutput(asr=asr_res, translation=tr_res, tts=tts_res, timings=timings)
