"""Metric implementations.

Phase 0 ships the metrics that don't need heavy models: WER (ASR),
duration deviation (isochrony), real-time factor and latency (system).
Quality metrics that need models/refs (COMET, SECS, UTMOS) are stubbed with
clear NotImplemented markers to be filled when real stages land.
"""
from __future__ import annotations

from typing import List


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


# --- Quality metrics requiring models/refs: filled in later phases ---

def comet_score(*args, **kwargs) -> float:
    raise NotImplementedError("COMET: wire in Phase 1 with real translation refs.")


def speaker_similarity(*args, **kwargs) -> float:
    raise NotImplementedError("SECS: wire in Phase 1 with a speaker-embedding model.")


def utmos(*args, **kwargs) -> float:
    raise NotImplementedError("UTMOS: wire in Phase 1 with a naturalness estimator.")
