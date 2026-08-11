#!/usr/bin/env python3
"""Generate a synthetic talking-head test video.

We need a *video with a face and English speech* to exercise the video pipeline,
but FLEURS is audio-only and shipping a real person's face raises privacy/licence
issues. So draw a simple neutral face and attach a real FLEURS English clip as
the audio track. Good enough to test the video plumbing + DummyLipSync end to
end on any machine; the neural Wav2Lip path needs a real face (run on Kaggle).

Usage:
    python scripts/make_test_video.py                     # uses fleurs_en_us_1938.wav
    python scripts/make_test_video.py --audio path.wav --out data/samples/testvideo_en.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

from s2st.config import resolve_path
from s2st.video import write_video


def draw_face(size: int = 320) -> Image.Image:
    """A plain neutral face (closed mouth) — the lip-sync stage adds the mouth."""
    img = Image.new("RGB", (size, size), (30, 32, 40))
    d = ImageDraw.Draw(img)
    cx = size // 2
    d.ellipse([cx - 110, 40, cx + 110, 300], fill=(224, 189, 160))  # head
    d.ellipse([cx - 60, 130, cx - 20, 165], fill=(255, 255, 255))   # eyes
    d.ellipse([cx + 20, 130, cx + 60, 165], fill=(255, 255, 255))
    d.ellipse([cx - 46, 140, cx - 34, 155], fill=(40, 40, 40))
    d.ellipse([cx + 34, 140, cx + 46, 155], fill=(40, 40, 40))
    d.line([cx, 175, cx, 205], fill=(180, 140, 120), width=4)        # nose
    d.line([cx - 24, 232, cx + 24, 232], fill=(120, 70, 70), width=4)  # neutral mouth
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default="data/samples/fleurs_en_us_1938.wav")
    ap.add_argument("--out", default="data/samples/testvideo_en.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, default=320)
    args = ap.parse_args()

    import soundfile as sf

    apath = resolve_path(args.audio)
    audio, sr = sf.read(str(apath), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    n_frames = max(1, int(round(duration * args.fps)))

    face = np.asarray(draw_face(args.size), dtype=np.uint8)
    frames = np.repeat(face[None, ...], n_frames, axis=0)  # static face, real audio

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out), frames, args.fps, audio=audio, audio_sr=sr)
    print(f"wrote {out}  ({n_frames} frames @ {args.fps}fps, {duration:.1f}s, audio {sr}Hz)")


if __name__ == "__main__":
    main()
