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


def test_normalize_text():
    # casing + punctuation should not count as errors
    assert metrics.normalize_text("Hello, World!") == "hello world"
    assert metrics.wer(
        metrics.normalize_text("Hello, world."),
        metrics.normalize_text("hello world"),
    ) == 0.0


def test_fit_duration_noop():
    # no-op paths must not require librosa (early returns)
    from s2st.audio import fit_duration

    a = np.zeros(24000, dtype=np.float32)  # 1 s at 24 kHz
    out, f = fit_duration(a, 24000, 0.0)   # target <= 0 -> unchanged
    assert f == 1.0 and len(out) == len(a)
    out, f = fit_duration(a, 24000, 1.0)   # already matches -> no stretch
    assert abs(f - 1.0) < 1e-6 and len(out) == len(a)


def test_factory_builds_dummy_pipeline():
    from s2st.factory import build_pipeline

    cfg = {
        "stages": {"asr": "dummy", "translation": "dummy", "tts": "dummy"},
        "language": {"src": "en", "tgt": "hi"},
        "audio": {"sample_rate": 16000},
    }
    pipe = build_pipeline(cfg)
    out = pipe.run(np.zeros(16000, dtype=np.float32), 16000, tgt_lang="hi")
    assert out.translation.tgt_lang == "hi"
    assert out.tts.audio.size > 0


def test_rtf():
    assert metrics.real_time_factor(0.5, 1.0) == 0.5


def test_harness_wires_secs_utmos():
    # SECS/UTMOS are computed only when enabled, and must flow into ItemResult.
    # Stub the (heavy) metric fns so the plumbing is tested without models.
    import json
    import os
    import tempfile

    from s2st.eval import evaluate
    from s2st.factory import build_pipeline

    cfg = {
        "stages": {"asr": "dummy", "translation": "dummy", "tts": "dummy"},
        "language": {"src": "en", "tgt": "hi"},
        "audio": {"sample_rate": 16000},
    }
    pipe = build_pipeline(cfg)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump([{"id": "t1", "ref_tgt": "x", "tgt_lang": "hi"}], f)
        manifest = f.name

    orig_secs, orig_utmos = metrics.speaker_similarity, metrics.utmos
    try:
        metrics.speaker_similarity = lambda *a, **k: 0.83
        metrics.utmos = lambda *a, **k: 3.9

        off = evaluate(pipe, manifest, sample_rate=16000)
        assert off[0].secs is None and off[0].utmos is None

        on = evaluate(pipe, manifest, sample_rate=16000, enable_secs=True, enable_utmos=True)
        assert abs(on[0].secs - 0.83) < 1e-9
        assert abs(on[0].utmos - 3.9) < 1e-9
    finally:
        metrics.speaker_similarity = orig_secs
        metrics.utmos = orig_utmos
        os.unlink(manifest)


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
