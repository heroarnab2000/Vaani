"""Metric implementations.

Phase 0 ships the metrics that don't need heavy models: WER (ASR),
duration deviation (isochrony), real-time factor and latency (system).
Quality metrics that need models/refs (COMET, SECS, UTMOS) are stubbed with
clear NotImplemented markers to be filled when real stages land.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

import numpy as np

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (Unicode-aware).

    Used before WER so Whisper's cased/punctuated output is compared fairly
    against normalized references. \\w keeps Devanagari, so it works for Hindi
    too.
    """
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance over word tokens.

    Returns errors / reference_words. Lower is better. Pure-Python so the
    harness has zero heavy deps in Phase 0.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0

    # classic DP edit distance
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
    return d[len(ref)][len(hyp)] / len(ref)


def duration_deviation(target_budget: float, realized: float) -> float:
    """Absolute fractional deviation of realized speech length from budget.

    This is the isochrony metric. 0.0 == perfect; 0.15 == 15% off.
    """
    if target_budget <= 0:
        return 0.0
    return abs(realized - target_budget) / target_budget


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float:
    """RTF < 1.0 means faster than real time."""
    if audio_seconds <= 0:
        return float("inf")
    return processing_seconds / audio_seconds


# --- Translation quality (Phase 1) ---

def bleu(references: List[str], hypotheses: List[str], tokenize: str = "flores200") -> float:
    """Corpus BLEU over parallel ref/hyp lists, as a 0-100 score.

    Defaults to the flores200 tokenizer (a.k.a. spBLEU) -- the standard for
    FLEURS / NLLB and the right choice for Hindi, where the default 13a
    tokenizer is unreliable. Falls back to 13a if the flores200 SPM is
    unavailable.
    """
    import sacrebleu

    try:
        score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    except Exception:
        score = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="13a")
    return float(score.score)


_COMET_MODEL = None


def comet_score(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    gpus: int = 0,
) -> float:
    """System-level COMET (reference-based). Loads + caches the model on first
    call. CPU by default (gpus=0) to avoid contending for scarce VRAM."""
    global _COMET_MODEL
    from comet import download_model, load_from_checkpoint

    if _COMET_MODEL is None:
        _COMET_MODEL = load_from_checkpoint(download_model(model_name))
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    out = _COMET_MODEL.predict(data, batch_size=8, gpus=gpus, progress_bar=False)
    return float(out["system_score"])


# --- TTS quality (Phase 1 metrics, wired here) ---

_SECS_MODEL = None
_UTMOS_MODEL = None


def speaker_similarity(
    reference: np.ndarray,
    generated: np.ndarray,
    ref_sr: int,
    gen_sr: int,
    model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
) -> float:
    """SECS: cosine similarity of ECAPA-TDNN speaker embeddings.

    Measures whether the synthesized clip keeps the *source speaker's* voice --
    the whole point of the voice-preserving TTS. ``reference`` is the source
    audio (what XTTS cloned from), ``generated`` is the TTS output. Both are
    resampled to 16 kHz (what ECAPA expects) and embedded; the return is their
    cosine similarity, where 1.0 == identical embedding and voice clones
    typically land ~0.7-0.9.

    The ECAPA encoder is cached on first call and runs on CPU to avoid
    contending for the scarce 4 GB VRAM. Deliberately uses a *different* encoder
    than XTTS's internal one, so the score isn't self-referential.
    """
    global _SECS_MODEL
    import torch

    try:  # speechbrain >= 1.0
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:  # older layout
        from speechbrain.pretrained import EncoderClassifier

    from ..audio import resample_linear, to_mono

    if _SECS_MODEL is None:
        _SECS_MODEL = EncoderClassifier.from_hparams(
            source=model_name, run_opts={"device": "cpu"}
        )

    def _embed(wav: "np.ndarray", sr: int) -> "torch.Tensor":
        w = resample_linear(to_mono(np.asarray(wav, dtype=np.float32)), sr, 16000)
        t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            return _SECS_MODEL.encode_batch(t).reshape(-1)

    e_ref = _embed(reference, ref_sr)
    e_gen = _embed(generated, gen_sr)
    return float(torch.nn.functional.cosine_similarity(e_ref, e_gen, dim=0))


def utmos(audio: np.ndarray, sample_rate: int) -> float:
    """UTMOS: predicted naturalness MOS (UTMOS22-strong), reference-free.

    Runs the SpeechMOS UTMOS22-strong predictor (tarepan/SpeechMOS, loaded via
    torch.hub -- no extra pip dep) on the TTS output, resampled to the 16 kHz
    mono the model expects. Higher is better (roughly a 1-5 MOS scale).
    Reference-free -- it scores the generated clip alone. Model cached on first
    call and run on CPU.
    """
    global _UTMOS_MODEL
    import torch

    from ..audio import resample_linear, to_mono

    if _UTMOS_MODEL is None:
        _UTMOS_MODEL = torch.hub.load(
            "tarepan/SpeechMOS", "utmos22_strong", trust_repo=True
        )

    wav = resample_linear(to_mono(np.asarray(audio, dtype=np.float32)), sample_rate, 16000)
    t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        score = _UTMOS_MODEL(t, 16000)
    return float(score.reshape(-1)[0])
