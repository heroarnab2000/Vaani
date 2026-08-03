#!/usr/bin/env python3
"""Run the pipeline over a manifest and print the metric table.

Stages are chosen by configs/default.yaml (or --config); nothing here is
hardcoded to dummy/real, so the same entry point serves every phase.

Usage:
    python scripts/run_eval.py                       # default config + manifest
    python scripts/run_eval.py --config configs/default.yaml
    python scripts/run_eval.py path/to/manifest.json # override manifest
"""
import argparse
import sys
from pathlib import Path

# print Hindi (Devanagari) safely regardless of the console code page
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from s2st.config import load_config, resolve_path
from s2st.factory import build_pipeline
from s2st.eval import evaluate, print_report, translation_corpus_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="manifest JSON (overrides config)")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--isochrony", choices=["on", "off"], default=None,
        help="override isochrony.enabled (for the on/off ablation)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.isochrony is not None:
        cfg.setdefault("isochrony", {})["enabled"] = args.isochrony == "on"
    pipeline = build_pipeline(cfg)

    manifest = args.manifest or cfg.get("eval", {}).get(
        "manifest", "data/samples/manifest.example.json"
    )
    manifest_path = str(resolve_path(manifest))

    sample_rate = cfg.get("audio", {}).get("sample_rate", 16000)
    results = evaluate(pipeline, manifest_path, sample_rate=sample_rate)

    eval_cfg = cfg.get("eval", {}) or {}
    tmetrics = translation_corpus_metrics(
        results,
        enable_comet=bool(eval_cfg.get("comet", False)),
        comet_gpus=int(eval_cfg.get("comet_gpus", 0)),
    )
    print_report(results, translation=tmetrics)


if __name__ == "__main__":
    main()
