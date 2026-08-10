"""Lip-sync plumbing tests: loudness envelope, DummyLipSync, video I/O round-trip.

The video/PIL deps (imageio, imageio-ffmpeg, Pillow) are NOT in the CI light-core,
so the tests that need them skip cleanly there and run locally / on the GPU box.
The envelope test is pure numpy and always runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def _have(*mods) -> bool:
    import importlib

    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            return False
    return True


def test_loudness_envelope():
    from s2st.lipsync.dummy import _loudness_envelope

    sr, fps, n = 16000, 25, 10
    silent = _loudness_envelope(np.zeros(sr, dtype=np.float32), sr, fps, n)
    assert silent.shape == (n,) and float(silent.max()) == 0.0  # silence -> closed mouth

    env = _loudness_envelope(np.ones(sr, dtype=np.float32), sr, fps, n)
    assert env.min() >= 0.0 and abs(env.max() - 1.0) < 1e-6  # normalized to peak 1


def test_dummy_lipsync_shapes():
    if not _have("PIL"):
        print("SKIP test_dummy_lipsync_shapes (no Pillow)")
        return
    from s2st.lipsync.dummy import DummyLipSync

    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)  # 8 source frames
    audio = (np.random.RandomState(0).randn(16000) * 0.1).astype(np.float32)  # 1.0s
    res = DummyLipSync().sync(frames, 25.0, audio, 16000)
    assert res.frames.shape[1:] == frames.shape[1:] and res.frames.dtype == np.uint8
    assert len(res.frames) == 25  # output length follows the audio (1.0s @ 25fps)
    assert res.frames.sum() > 0   # a mouth got drawn onto the blank frames


def test_video_io_roundtrip():
    if not _have("imageio", "imageio_ffmpeg", "PIL", "soundfile"):
        print("SKIP test_video_io_roundtrip (video deps not installed)")
        return
    import os
    import tempfile

    from s2st.video import extract_audio, read_frames, write_video

    frames = (np.random.RandomState(1).rand(25, 48, 48, 3) * 255).astype(np.uint8)  # 1.0s
    audio = (np.sin(np.linspace(0, 200, 16000)) * 0.2).astype(np.float32)           # 1.0s
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mp4")
        write_video(p, frames, 25.0, audio, 16000)
        assert os.path.exists(p)
        rf, fps = read_frames(p)
        assert len(rf) >= 22  # frame count ~preserved (codec may pad/drop a couple)
        a = extract_audio(p, 16000)
        assert a is not None and len(a) > 8000


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
