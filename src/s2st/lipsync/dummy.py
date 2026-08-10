"""DummyLipSync — amplitude-driven mouth overlay.

A crude but *real* lip-sync with zero heavy deps: it opens/closes a drawn mouth
in step with the audio's loudness envelope. It does NOT detect a face (it draws
at a fixed lower-centre position), so it's for **testing the video plumbing and
the interface end-to-end on any machine** — not for a believable result. The
believable version is Wav2LipStage (neural, GPU). Both satisfy LipSyncStage, so
swapping them is a one-line config change.
"""
from __future__ import annotations

import numpy as np

from .base import LipSyncResult, LipSyncStage


def _loudness_envelope(audio: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """Per-frame RMS loudness in [0, 1], one value per video frame."""
    if audio is None or np.asarray(audio).size == 0 or n_frames <= 0:
        return np.zeros(max(n_frames, 0), dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    hop = max(1, int(round(sr / max(fps, 1e-6))))
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = audio[i * hop : (i + 1) * hop]
        if chunk.size:
            env[i] = np.sqrt(np.mean(np.square(chunk)))
    peak = float(env.max())
    # below ~-80 dBFS the whole clip is silence -> flat closed mouth, not gaping
    return env / peak if peak > 1e-4 else np.zeros(n_frames, dtype=np.float32)


class DummyLipSync(LipSyncStage):
    def sync(self, frames, fps, audio, audio_sr) -> LipSyncResult:
        from PIL import Image, ImageDraw

        frames = np.asarray(frames, dtype=np.uint8)
        audio = np.asarray(audio, dtype=np.float32)
        # Match output length to the audio, not the input video (Wav2Lip does the
        # same — it loops the face frames to cover the whole audio). Otherwise a
        # longer translation would get truncated on mux.
        n = max(1, int(round(len(audio) / audio_sr * fps))) if audio.size else len(frames)
        src = frames[np.arange(n) % len(frames)]
        env = _loudness_envelope(audio, audio_sr, fps, n)
        h, w = frames[0].shape[:2]
        cx, cy = w // 2, int(h * 0.72)          # fixed lower-centre "mouth" position
        half_w = max(4, int(0.16 * w))

        out = np.empty_like(src)
        for i in range(n):
            img = Image.fromarray(src[i])
            draw = ImageDraw.Draw(img)
            open_px = int(2 + env[i] * 0.14 * h)  # mouth opening tracks loudness
            draw.ellipse(
                [cx - half_w, cy - open_px, cx + half_w, cy + open_px],
                fill=(70, 25, 25),
                outline=(30, 10, 10),
            )
            out[i] = np.asarray(img, dtype=np.uint8)
        return LipSyncResult(frames=out, fps=fps)
