"""Lip-sync stage: given face video frames + target-language audio, return
frames whose mouth matches that audio.

Mirrors the ASR/MT/TTS pattern: one abstract interface, a zero-dependency dummy
for testing the video plumbing on any machine, and a real model (Wav2Lip) that
runs on a GPU. The orchestrator/scripts only ever touch this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class LipSyncResult:
    frames: np.ndarray  # [T, H, W, 3] uint8, mouth re-synced to the audio
    fps: float


class LipSyncStage(ABC):
    @abstractmethod
    def sync(
        self,
        frames: np.ndarray,
        fps: float,
        audio: np.ndarray,
        audio_sr: int,
    ) -> LipSyncResult:
        """Re-draw mouths in ``frames`` to match ``audio``."""


def build_lipsync(cfg: dict) -> LipSyncStage:
    """Config -> concrete lip-sync stage (lazy heavy imports, like the factory)."""
    name = (cfg.get("stages", {}) or {}).get("lipsync", "dummy")
    if name == "dummy":
        from .dummy import DummyLipSync

        return DummyLipSync()
    if name == "wav2lip":
        from .wav2lip import Wav2LipStage

        m = (cfg.get("models", {}) or {}).get("lipsync", {}) or {}
        return Wav2LipStage(
            repo_dir=m.get("repo_dir", "Wav2Lip"),
            checkpoint=m.get("checkpoint", "Wav2Lip/checkpoints/wav2lip_gan.pth"),
            device=m.get("device", "cuda"),
        )
    raise ValueError(f"unknown lipsync stage: {name!r}")
