#!/usr/bin/env python3
"""Video in -> translated, lip-synced video out (the D1 demo).

Pipeline: read video -> extract English audio -> S2ST cascade (ASR->MT->TTS) ->
Hindi audio in the speaker's voice -> lip-sync the frames to that Hindi audio ->
mux -> output MP4.

Stages come from the config, so this runs on dummy stages locally (fast, proves
plumbing) and on real models + Wav2Lip on a GPU by swapping the config. See
docs/LIPSYNC_KAGGLE.md for the GPU path.

Usage:
    python scripts/lipsync_demo.py --config configs/dummy.yaml   # local plumbing
    python scripts/lipsync_demo.py data/samples/testvideo_en.mp4 --config configs/gpu_lipsync.yaml
"""
import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from s2st.config import load_config, resolve_path
from s2st.factory import build_pipeline
from s2st.lipsync import build_lipsync
from s2st.video import extract_audio, read_frames, write_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?", default="data/samples/testvideo_en.mp4")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/samples/testvideo_hi_lipsync.mp4")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sr = cfg.get("audio", {}).get("sample_rate", 16000)
    vpath = resolve_path(args.video)

    frames, fps = read_frames(str(vpath))
    audio = extract_audio(str(vpath), sr)
    if audio is None:
        raise SystemExit(f"no audio track in {vpath}")
    print(f"in : {len(frames)} frames @ {fps:.0f}fps, {len(audio)/sr:.1f}s audio")

    # English audio -> Hindi audio in the speaker's voice
    pipe = build_pipeline(cfg)
    out = pipe.run(audio, sr, tgt_lang=cfg.get("language", {}).get("tgt", "hi"))
    print(f"ASR: {out.asr.text[:70]}")
    print(f"MT : {out.translation.text[:70]}")

    # re-sync the mouth to the translated audio
    lip = build_lipsync(cfg)
    synced = lip.sync(frames, fps, out.tts.audio, out.tts.sample_rate)

    outp = resolve_path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(outp), synced.frames, synced.fps, out.tts.audio, out.tts.sample_rate)
    print(f"out: {outp}  ({len(synced.frames)} frames, {out.tts.realized_duration:.1f}s Hindi audio)")


if __name__ == "__main__":
    main()
