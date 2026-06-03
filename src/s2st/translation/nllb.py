"""NLLB-200 translation stage (Phase 1).

Text-to-text MT with NLLB-200 (distilled 600M) via transformers. Establishes
the BLEU/COMET baseline.

The isochrony duration budget (`target_speech_duration`) is threaded through
and recorded but does NOT yet influence generation -- that is the Phase 2
contribution. Keeping the plumbing here means Phase 2 only changes *how* we
decode, not the pipeline.

Selected via configs/default.yaml: `stages.translation: nllb`.
"""
from __future__ import annotations

from typing import Optional

from ..interfaces import TranslationStage
from ..types import ASRResult, TranslationResult

# NLLB uses FLORES-200 language codes.
_NLLB_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
}


class NLLBTranslation(TranslationStage):
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        src_lang: str = "en",
        num_beams: int = 5,
        max_new_tokens: int = 256,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.src_lang = src_lang
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def _code(lang: str) -> str:
        return _NLLB_CODES.get(lang, lang)

    def translate(
        self,
        asr: ASRResult,
        tgt_lang: str,
        target_speech_duration: float,
    ) -> TranslationResult:
        src_lang = asr.language or self.src_lang
        text_out = ""
        if asr.text.strip():
            self.tokenizer.src_lang = self._code(src_lang)
            inputs = self.tokenizer(asr.text, return_tensors="pt").to(self.device)
            bos = self.tokenizer.convert_tokens_to_ids(self._code(tgt_lang))
            with self._torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    forced_bos_token_id=bos,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )
            text_out = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

        return TranslationResult(
            text=text_out,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            target_speech_duration=target_speech_duration,
            n_candidates=1,
            chosen_length_bucket="baseline",  # no length control yet (Phase 2)
        )
