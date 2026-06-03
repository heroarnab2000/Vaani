"""Evaluation harness (Phase 0 deliverable).

Runs the pipeline over a manifest of test items and prints a metric table.
With the dummy stages the numbers are meaningless on purpose -- the point of
Phase 0 is to prove the *plumbing* works: every metric computes, aggregates,
and prints. Real stages later flow through unchanged.

Manifest format (JSON list):
    [{"audio": "data/samples/x.wav", "ref_src": "...", "ref_tgt": "...",
      "tgt_lang": "hi"}, ...]
If "audio" is missing/unreadable, a synthetic tone is used so the harness
still runs in CI with no data files.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..pipeline.orchestrator import S2STPipeline
from . import metrics


@dataclass
class ItemResult:
    item_id: str
    wer: Optional[float]
    duration_deviation: float
    rtf: float
    latency: float
    # raw text kept for corpus-level translation metrics (BLEU/COMET)
    src_text: str = ""
    hyp_tgt: str = ""
    ref_tgt: str = ""


def _load_audio(path: Optional[str], sample_rate: int = 16000) -> np.ndarray:
    """Load a wav as mono float32 at sample_rate, or synthesize 3s of tone if
    unavailable (keeps the harness runnable with no data files)."""
    if path:
        from ..audio import load_wav
        from ..config import resolve_path

        full = resolve_path(path)
        if full.exists():
            audio = load_wav(full, sample_rate)
            if audio is not None and audio.size:
                return audio
    # fallback synthetic audio
    t = np.linspace(0, 3.0, sample_rate * 3, endpoint=False)
    return (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def evaluate(
    pipeline: S2STPipeline,
    manifest_path: str,
    sample_rate: int = 16000,
) -> List[ItemResult]:
    items = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    results: List[ItemResult] = []

    for i, item in enumerate(items):
        audio = _load_audio(item.get("audio"), sample_rate)
        out = pipeline.run(
            audio,
            sample_rate,
            tgt_lang=item.get("tgt_lang", "hi"),
        )

        ref_src = item.get("ref_src")
        item_wer = (
            metrics.wer(metrics.normalize_text(ref_src), metrics.normalize_text(out.asr.text))
            if ref_src
            else None
        )

        dev = metrics.duration_deviation(
            out.translation.target_speech_duration,
            out.tts.realized_duration,
        )
        rtf = metrics.real_time_factor(out.total_latency, out.asr.audio_duration)

        results.append(
            ItemResult(
                item_id=item.get("id", f"item{i}"),
                wer=item_wer,
                duration_deviation=dev,
                rtf=rtf,
                latency=out.total_latency,
                src_text=out.asr.text,
                hyp_tgt=out.translation.text,
                ref_tgt=item.get("ref_tgt", "") or "",
            )
        )
    return results


def translation_corpus_metrics(
    results: List[ItemResult],
    enable_comet: bool = False,
    comet_gpus: int = 0,
) -> dict:
    """Corpus-level translation quality over items that have a target reference.

    BLEU is always computed (cheap); COMET only if enabled (heavy model load).
    """
    triples = [(r.src_text, r.hyp_tgt, r.ref_tgt) for r in results if r.ref_tgt and r.hyp_tgt]
    if not triples:
        return {}
    srcs, hyps, refs = (list(x) for x in zip(*triples))
    out: dict = {"items_with_ref": len(triples)}
    try:
        out["BLEU"] = metrics.bleu(refs, hyps)
    except ImportError:
        pass  # sacrebleu not installed yet (dummy translation -> BLEU not meaningful)
    if enable_comet:
        try:
            out["COMET"] = metrics.comet_score(srcs, hyps, refs, gpus=comet_gpus)
        except Exception as e:  # noqa: BLE001
            out["COMET"] = None
            out["COMET_error"] = str(e)[:200]
    return out


def _agg(values: List[float]) -> str:
    vals = [v for v in values if v is not None and v != float("inf")]
    if not vals:
        return "n/a"
    med = statistics.median(vals)
    mean = statistics.fmean(vals)
    return f"mean={mean:.3f} median={med:.3f}"


def print_report(results: List[ItemResult], translation: Optional[dict] = None) -> None:
    print("\n=== S2ST evaluation report ===")
    print(f"items: {len(results)}\n")
    header = f"{'id':<14} {'WER':>8} {'dur_dev':>8} {'RTF':>8} {'lat(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        wer_s = f"{r.wer:.3f}" if r.wer is not None else "n/a"
        print(
            f"{r.item_id:<14} {wer_s:>8} {r.duration_deviation:>8.3f} "
            f"{r.rtf:>8.3f} {r.latency:>8.3f}"
        )
    print("-" * len(header))
    print(f"{'WER':>19}: {_agg([r.wer for r in results])}")
    print(f"{'duration_deviation':>19}: {_agg([r.duration_deviation for r in results])}")
    print(f"{'RTF':>19}: {_agg([r.rtf for r in results])}")
    print(f"{'latency':>19}: {_agg([r.latency for r in results])}")
    if translation:
        n = translation.get("items_with_ref", "?")
        print(f"\n-- translation quality (n={n}) --")
        if "BLEU" in translation:
            print(f"{'spBLEU':>19}: {translation['BLEU']:.2f}")
        if translation.get("COMET") is not None:
            print(f"{'COMET':>19}: {translation['COMET']:.4f}")
        elif "COMET_error" in translation:
            print(f"{'COMET':>19}: error ({translation['COMET_error']})")
    print("==============================\n")
