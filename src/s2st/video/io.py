"""Video I/O — read/write frames and extract/mux audio.

Dependency-light: `imageio` for the frame stream and `imageio-ffmpeg`'s bundled
static ffmpeg for audio (no system ffmpeg needed). Frames are uint8 arrays of
shape [T, H, W, 3] (RGB). This is the video counterpart to s2st.audio, and the
lip-sync stages read/write through it so the model code never touches ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def read_frames(path: str) -> Tuple[np.ndarray, float]:
    """Load a video as (frames [T,H,W,3] uint8, fps)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    fps = float(reader.get_meta_data().get("fps", 25.0))
    frames = np.stack([np.asarray(f)[:, :, :3] for f in reader])
    reader.close()
    return frames.astype(np.uint8), fps


def extract_audio(path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Extract the audio track as mono float32 at ``sample_rate``.

    Returns None if the video has no audio track (ffmpeg exits non-zero).
    """
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        wav = str(Path(d) / "audio.wav")
        proc = subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", str(path), "-vn",
             "-ac", "1", "-ar", str(sample_rate), wav],
            capture_output=True,
        )
        if proc.returncode != 0 or not Path(wav).exists():
            return None
        audio, _ = sf.read(wav, dtype="float32")
    return np.ascontiguousarray(audio, dtype=np.float32)


def write_video(
    path: str,
    frames: np.ndarray,
    fps: float,
    audio: Optional[np.ndarray] = None,
    audio_sr: int = 16000,
) -> None:
    """Write frames (and optional audio) to an MP4 (H.264 / AAC).

    Frames are encoded silently first, then the audio is muxed in with ffmpeg so
    the two stay independent (the whole point of the lip-sync stage: video comes
    from the sync model, audio from the TTS).
    """
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as d:
        silent = str(Path(d) / "silent.mp4")
        writer = imageio.get_writer(
            silent, fps=fps, codec="libx264", quality=8, macro_block_size=1
        )
        for f in frames:
            writer.append_data(np.asarray(f, dtype=np.uint8))
        writer.close()

        if audio is None or np.asarray(audio).size == 0:
            shutil.copy(silent, str(path))
            return

        import soundfile as sf

        awav = str(Path(d) / "audio.wav")
        sf.write(awav, np.asarray(audio, dtype=np.float32), audio_sr)
        subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", silent, "-i", awav,
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)],
            check=True, capture_output=True,
        )
