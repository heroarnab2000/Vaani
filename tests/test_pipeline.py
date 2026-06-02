"""Tests that lock the Phase 0 contract: metrics compute, pipeline runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from s2st.asr import DummyASR
from s2st.translation import DummyTranslation
from s2st.tts import DummyTTS
from s2st.pipeline import S2STPipeline
from s2st.eval import metrics


def test_wer_perfect():
    assert metrics.wer("a b c", "a b c") == 0.0


def test_wer_all_wrong():
    assert metrics.wer("a b c", "x y z") == 1.0


def test_wer_one_sub():
    assert abs(metrics.wer("a b c", "a x c") - 1 / 3) < 1e-9


def test_duration_deviation():
    assert metrics.duration_deviation(2.0, 2.0) == 0.0
    assert abs(metrics.duration_deviation(2.0, 2.3) - 0.15) < 1e-9


def test_rtf():
    assert metrics.real_time_factor(0.5, 1.0) == 0.5


def test_pipeline_runs_end_to_end():
    pipe = S2STPipeline(DummyASR(), DummyTranslation(), DummyTTS())
    audio = np.zeros(16000 * 2, dtype=np.float32)  # 2s silence
    out = pipe.run(audio, 16000, tgt_lang="hi")
    assert out.asr.text
    assert out.translation.tgt_lang == "hi"
    assert out.tts.audio.size > 0
    assert set(out.timings) == {"asr", "translation", "tts"}
    assert out.total_latency >= 0.0


if __name__ == "__main__":
    # tiny runner so `python tests/test_pipeline.py` works without pytest
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
