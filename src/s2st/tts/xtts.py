"""XTTS-v2 voice-preserving TTS stage (Phase 1).

Zero-shot voice cloning: synthesizes target-language speech in the *source*
speaker's voice from a few seconds of reference audio (the input clip). XTTS-v2
supports Hindi ("hi"). Native output sample rate is 24 kHz.

XTTS-v2 is released under the non-commercial CPML license; we set
COQUI_TOS_AGREED so the loader doesn't block on an interactive prompt.

Selected via configs/default.yaml: `stages.tts: xtts_v2`.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

from ..audio import resample_linear, to_mono
from ..interfaces import TTSStage
from ..types import TranslationResult, TTSResult

XTTS_SR = 24000  # XTTS-v2 native output sample rate
REF_SR = 16000   # reference audio resampled to 16 kHz for speaker conditioning

# XTTS's text cleaner expands ASCII digits via num2words, which has no Hindi
# support and raises NotImplementedError. Its number regexes are ASCII-only
# ([0-9]), so mapping digits to Devanagari numerals sidesteps expansion while
# leaving the MT output (and BLEU) untouched. NOTE: digit *verbalization* in
# Hindi TTS is a known gap to revisit (needs a Hindi number normalizer).
_HI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def _sanitize_for_xtts(text: str, lang: str) -> str:
    return text.translate(_HI_DIGITS) if lang == "hi" else text


class XTTSv2TTS(TTSStage):
    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cpu",
        tgt_lang: str = "hi",
    ):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")  # accept non-commercial license
        from TTS.api import TTS

        self.tts = TTS(model_name).to(device)
        self.tgt_lang = tgt_lang

    def synthesize(
        self,
        translation: TranslationResult,
        speaker_wav: Optional[np.ndarray],
        sample_rate: int,
    ) -> TTSResult:
        lang = translation.tgt_lang or self.tgt_lang
        text = _sanitize_for_xtts(translation.text.strip(), lang)
        if not text:
            return TTSResult(np.zeros(1, dtype=np.float32), XTTS_SR, 0.0)

        # XTTS clones from a reference wav *file*; write the source speaker audio
        # (mono, 16 kHz) to a temp file for conditioning.
        tmp_path = None
        try:
            if speaker_wav is not None and np.asarray(speaker_wav).size:
                import soundfile as sf

                ref16 = resample_linear(to_mono(speaker_wav), sample_rate, REF_SR)
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(tmp_path, ref16, REF_SR)
                wav = self.tts.tts(text=text, speaker_wav=tmp_path, language=lang)
            else:
                wav = self.tts.tts(text=text, language=lang)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        audio = np.asarray(wav, dtype=np.float32)
        return TTSResult(audio=audio, sample_rate=XTTS_SR, realized_duration=len(audio) / XTTS_SR)
