# Real lip-sync (Wav2Lip) on a Kaggle GPU

The dummy lip-sync (`DummyLipSync`) runs anywhere and proves the video plumbing,
but it just draws a mouth bar. For a believable result you need a neural model on
a GPU. This is the **offline "D1" demo**: a talking-head video in → the same
person appearing to speak Hindi out. It does **not** need streaming.

> Prereqs: the audio stack from [`KAGGLE.md`](KAGGLE.md) already installed, GPU +
> Internet on. Video deps: `pip install imageio imageio-ffmpeg pillow`.

## 1. Get Wav2Lip + its weights
```python
!git clone https://github.com/Rudrabha/Wav2Lip
# checkpoint (wav2lip_gan.pth) + the s3fd face detector weights.
# The original Google-Drive links rot often; fetch from a current mirror into:
#   Wav2Lip/checkpoints/wav2lip_gan.pth
#   Wav2Lip/face_detection/detection/sfd/s3fd.pth
```
> **This is the fiddly part.** Wav2Lip is old code (pins an ancient `librosa`
> that fights modern numpy). On Kaggle you typically need `pip install
> librosa==0.9.1 numba==0.58` (or run its inference in a separate env). If setup
> fights you, **MuseTalk** (https://github.com/TMElyralab/MuseTalk) is the modern,
> real-time-friendly alternative and slots into the same `LipSyncStage` interface
> — add a `MuseTalkStage` mirroring `wav2lip.py`.

## 2. A real test video (with a face)
The synthetic `scripts/make_test_video.py` clip has no *real* face, so Wav2Lip's
face detector will reject it. Use a real talking-head clip:
- **Upload** a short clip of a person facing the camera as a Kaggle Dataset, or
- record ~10s on your phone and upload it.

Put it at `data/samples/testvideo_en.mp4` (or pass its path to the demo).

## 3. Run the full video demo
```python
!python scripts/lipsync_demo.py data/samples/testvideo_en.mp4 \
    --config configs/gpu_lipsync.yaml \
    --out data/samples/testvideo_hi_lipsync.mp4
```
This runs: extract English audio → ASR→MT→TTS (Hindi in the speaker's voice,
isochrony on so it stays lip-alignable) → **Wav2Lip re-syncs the mouth to the
Hindi audio** → muxed MP4 out.

## How it fits the codebase
- `src/s2st/lipsync/base.py` — `LipSyncStage` interface + `build_lipsync(cfg)`.
- `src/s2st/lipsync/dummy.py` — the zero-dep mouth overlay (local/CI testing).
- `src/s2st/lipsync/wav2lip.py` — drives Wav2Lip's `inference.py` (this doc).
- `src/s2st/video/io.py` — frame read/write + audio extract/mux (bundled ffmpeg).
- `scripts/lipsync_demo.py` — the end-to-end wiring; stages come from the config.

Swapping dummy ↔ wav2lip is one line in the config (`stages.lipsync`). Everything
else — the video I/O, the audio cascade, the mux — is identical.

## Honest limits (for now)
- **Offline only.** This is the file-in/file-out demo. Real-time ≤5s (Phase 7)
  needs the streaming work first.
- **Voice quality** is still the Hindi-TTS weak spot (SECS ~0.31) — the lips will
  match, but the voice isn't fully convincing yet. Tracked in
  [`VIDEO_ROADMAP.md`](VIDEO_ROADMAP.md) (R6).
- **Frontal faces only** — Wav2Lip works on roughly head-on talking video.
