#!/usr/bin/env python3
"""Isochrony on/off ablation: run the cascade both ways, print one comparison.

This is the Phase-2 deliverable -- a data-backed before/after for the isochrony
work (duration-aware MT reranking + TTS rate control). Same models, same test
set; only `isochrony.enabled` flips.

Usage:
    python scripts/ablation.py                 # uses eval.manifest from config
    python scripts/ablation.py path/to.json    # a specific manifest
"""
from __future__ import annotations

import argparse
import copy
import gc
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from s2st.config import load_config, resolve_path
from s2st.eval import evaluate, translation_corpus_metrics
from s2st.factory import build_pipeline


def _med(xs):
    xs = [x for x in xs if x is not None and x != float("inf")]
    return statistics.median(xs) if xs else float("nan")


def _mean(xs):
    xs = [x for x in xs if x is not None and x != float("inf")]
    return statistics.fmean(xs) if xs else float("nan")


def run_once(cfg: dict, manifest: str, sr: int, enabled: bool):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("isochrony", {})["enabled"] = enabled
    pipe = build_pipeline(cfg)
    results = evaluate(pipe, manifest, sample_rate=sr)
    tm = translation_corpus_metrics(results)
    del pipe  # free models before building the other pipeline (4 GB / laptop RAM)
    gc.collect()
    return results, tm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?", help="manifest JSON (overrides config)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sr = cfg.get("audio", {}).get("sample_rate", 16000)
    manifest = str(resolve_path(args.manifest or cfg.get("eval", {}).get("manifest")))

    print("Running isochrony OFF (baseline)...")
    off, off_tm = run_once(cfg, manifest, sr, enabled=False)
    print("Running isochrony ON (duration-aware MT + TTS rate control)...")
    on, on_tm = run_once(cfg, manifest, sr, enabled=True)

    rows = [
        ("duration_deviation (median)", _med([r.duration_deviation for r in off]),
         _med([r.duration_deviation for r in on]), "lower"),
        ("duration_deviation (mean)", _mean([r.duration_deviation for r in off]),
         _mean([r.duration_deviation for r in on]), "lower"),
        ("WER (median)", _med([r.wer for r in off]), _med([r.wer for r in on]), "same"),
        ("spBLEU", off_tm.get("BLEU", float("nan")), on_tm.get("BLEU", float("nan")), "hold"),
        ("RTF (median)", _med([r.rtf for r in off]), _med([r.rtf for r in on]), "cost"),
    ]

    print(f"\n=== Isochrony ablation ({len(off)} items) ===")
    print(f"{'metric':<30}{'OFF':>10}{'ON':>10}{'goal':>8}")
    print("-" * 58)
    for name, a, b, goal in rows:
        print(f"{name:<30}{a:>10.3f}{b:>10.3f}{goal:>8}")
    print("-" * 58)
    print("goal: lower=better, hold=keep ~equal, cost=expected to rise (Phase 4)\n")


if __name__ == "__main__":
    main()
