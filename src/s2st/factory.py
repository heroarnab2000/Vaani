"""Build pipeline stages from a config dict.

The orchestrator and eval harness only ever touch the abstract interfaces;
this factory is the single place that maps config strings -> concrete classes.
Real-model imports are lazy (inside each branch) so the dummy path never
requires the heavy deps (faster-whisper, transformers, coqui-tts) to be
installed.
"""
from __future__ import annotations

from typing import Any

from .interfaces import ASRStage, TranslationStage, TTSStage
from .pipeline import S2STPipeline


def build_asr(cfg: dict[str, Any]) -> ASRStage:
    name = cfg.get("stages", {}).get("asr", "dummy")
    m = cfg.get("models", {}).get("asr", {}) or {}
    src_lang = cfg.get("language", {}).get("src", "en")

    if name == "dummy":
        from .asr import DummyASR

        return DummyASR(language=src_lang)
    if name == "faster_whisper":
        from .asr.faster_whisper import FasterWhisperASR

        return FasterWhisperASR(
            model_name=m.get("name", "large-v3"),
            device=m.get("device", "cpu"),
            compute_type=m.get("compute_type", "int8"),
            language=src_lang,
        )
    raise ValueError(f"unknown asr stage: {name!r}")


def build_translation(cfg: dict[str, Any]) -> TranslationStage:
    name = cfg.get("stages", {}).get("translation", "dummy")
    tgt_lang = cfg.get("language", {}).get("tgt", "hi")

    if name == "dummy":
        from .translation import DummyTranslation

        return DummyTranslation(default_tgt=tgt_lang)
    if name == "nllb":
        from .translation.nllb import NLLBTranslation

        m = cfg.get("models", {}).get("translation", {}) or {}
        return NLLBTranslation(
            model_name=m.get("name", "facebook/nllb-200-distilled-600M"),
            device=m.get("device", "cpu"),
            src_lang=cfg.get("language", {}).get("src", "en"),
        )
    raise ValueError(f"unknown translation stage: {name!r}")


def build_tts(cfg: dict[str, Any]) -> TTSStage:
    name = cfg.get("stages", {}).get("tts", "dummy")
    audio = cfg.get("audio", {}) or {}

    if name == "dummy":
        from .tts import DummyTTS

        return DummyTTS(sample_rate=audio.get("sample_rate", 16000))
    if name == "xtts_v2":
        from .tts.xtts import XTTSv2TTS

        m = cfg.get("models", {}).get("tts", {}) or {}
        return XTTSv2TTS(
            model_name=m.get("name", "tts_models/multilingual/multi-dataset/xtts_v2"),
            device=m.get("device", "cpu"),
            tgt_lang=cfg.get("language", {}).get("tgt", "hi"),
        )
    raise ValueError(f"unknown tts stage: {name!r}")


def build_pipeline(cfg: dict[str, Any]) -> S2STPipeline:
    return S2STPipeline(
        asr=build_asr(cfg),
        translation=build_translation(cfg),
        tts=build_tts(cfg),
    )
