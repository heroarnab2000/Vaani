#!/usr/bin/env python3
"""Phase 0 smoke run: dummy pipeline -> eval harness -> metric table.

Usage:
    python scripts/run_eval.py [manifest.json]

With no argument it uses data/samples/manifest.example.json. This is the
Phase 0 checkpoint: it must print a full metric table with the dummy stages.
"""
import sys
from pathlib import Path

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from s2st.asr import DummyASR
from s2st.translation import DummyTranslation
from s2st.tts import DummyTTS
from s2st.pipeline import S2STPipeline
from s2st.eval import evaluate, print_report


def main() -> None:
    manifest = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).resolve().parent.parent
                 / "data/samples/manifest.example.json")
    )

    pipeline = S2STPipeline(
        asr=DummyASR(language="en"),
        translation=DummyTranslation(default_tgt="hi"),
        tts=DummyTTS(),
    )

    results = evaluate(pipeline, manifest)
    print_report(results)


if __name__ == "__main__":
    main()
