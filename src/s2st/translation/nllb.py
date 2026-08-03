"""NLLB-200 translation stage (Phase 1 baseline + Phase 2 isochrony).

Text-to-text MT with NLLB-200 (distilled 600M) via transformers.

Phase 2 adds *duration-aware decoding*: when `rerank` is on, we generate several
candidates with different length penalties (biasing shorter/longer outputs),
estimate each one's spoken duration from a Hindi chars-per-second prior, and keep
the candidate closest to the source-speech duration budget. This shrinks the
length gap the TTS rate control then has to absorb. With `rerank` off the stage
is the plain single-best baseline.

Selected via configs/default.yaml: `stages.translation: nllb`.
"""
from __future__ import annotations

from typing import List

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
        rerank: bool = False,
        n_candidates: int = 3,
        chars_per_sec: float = 14.0,
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
        self.rerank = rerank
        self.n_candidates = max(1, n_candidates)
        self.chars_per_sec = chars_per_sec

    @staticmethod
    def _code(lang: str) -> str:
        return _NLLB_CODES.get(lang, lang)

    def _length_penalties(self) -> List[float]:
        """Length penalties spanning short->long; 1 candidate == plain decode."""
        if self.n_candidates <= 1:
            return [1.0]
        lo, hi = 0.4, 1.8  # <1 favors shorter sequences, >1 favors longer
        step = (hi - lo) / (self.n_candidates - 1)
        return [round(lo + i * step, 3) for i in range(self.n_candidates)]

    def _decode(self, inputs, bos: int, length_penalty: float) -> str:
        with self._torch.no_grad():
            gen = self.model.generate(
                **inputs,
                forced_bos_token_id=bos,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                length_penalty=length_penalty,
            )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def translate(
        self,
        asr: ASRResult,
        tgt_lang: str,
        target_speech_duration: float,
    ) -> TranslationResult:
        src_lang = asr.language or self.src_lang
        text_out = ""
        n_used = 1
        bucket = "baseline"

        if asr.text.strip():
            self.tokenizer.src_lang = self._code(src_lang)
            inputs = self.tokenizer(asr.text, return_tensors="pt").to(self.device)
            bos = self.tokenizer.convert_tokens_to_ids(self._code(tgt_lang))

            if not self.rerank or target_speech_duration <= 0:
                text_out = self._decode(inputs, bos, length_penalty=1.0)
            else:
                # duration-aware decoding: generate length variants, keep the
                # candidate whose predicted spoken length best fits the budget.
                seen = {}
                for lp in self._length_penalties():
                    cand = self._decode(inputs, bos, length_penalty=lp)
                    if cand and cand not in seen:
                        seen[cand] = lp
                n_used = len(seen)
                target_chars = target_speech_duration * self.chars_per_sec
                text_out, lp = min(
                    seen.items(), key=lambda kv: abs(len(kv[0]) - target_chars)
                )
                bucket = f"lp={lp}"

        return TranslationResult(
            text=text_out,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            target_speech_duration=target_speech_duration,
            n_candidates=n_used,
            chosen_length_bucket=bucket,
        )
