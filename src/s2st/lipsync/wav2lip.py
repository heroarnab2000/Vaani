"""Wav2LipStage — real neural lip-sync (runs on a GPU).

Wav2Lip ships as a repo with an `inference.py`, not a pip package, so this stage
drives that script: it writes the frames + audio to temp files, runs inference,
and reads the re-synced video back. The heavy model therefore stays entirely
external — this wrapper imports nothing heavy, so the dummy path (and CI) never
needs Wav2Lip installed.

Setup (Kaggle / any GPU box) is in docs/LIPSYNC_KAGGLE.md:
  git clone https://github.com/Rudrabha/Wav2Lip
  + download the wav2lip_gan.pth checkpoint and the s3fd face-detector weights.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .base import LipSyncResult, LipSyncStage


class Wav2LipStage(LipSyncStage):
    def __init__(
        self,
        repo_dir: str = "Wav2Lip",
        checkpoint: str = "Wav2Lip/checkpoints/wav2lip_gan.pth",
        device: str = "cuda",
    ):
        self.repo_dir = Path(repo_dir)
        self.checkpoint = Path(checkpoint)
        self.device = device

    def sync(self, frames, fps, audio, audio_sr) -> LipSyncResult:
        from ..video.io import read_frames, write_video

        inference = self.repo_dir / "inference.py"
        if not inference.exists():
            raise FileNotFoundError(
                f"Wav2Lip not found at {inference}. Clone https://github.com/Rudrabha/Wav2Lip "
                f"and fetch the checkpoint — see docs/LIPSYNC_KAGGLE.md."
            )

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            face_mp4, out_mp4 = str(d / "face.mp4"), str(d / "out.mp4")
            write_video(face_mp4, frames, fps)  # silent; Wav2Lip drives audio itself
            awav = str(d / "audio.wav")
            import soundfile as sf

            sf.write(awav, np.asarray(audio, dtype=np.float32), audio_sr)

            subprocess.run(
                [sys.executable, "inference.py",
                 "--checkpoint_path", str(Path(self.checkpoint).resolve()),
                 "--face", face_mp4, "--audio", awav, "--outfile", out_mp4],
                cwd=str(self.repo_dir), check=True,
            )
            synced, out_fps = read_frames(out_mp4)
        return LipSyncResult(frames=synced, fps=out_fps)
