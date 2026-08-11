# Real lip-sync (Wav2Lip) on a Kaggle GPU — turnkey

The dummy path proves the plumbing; this produces the **real** result: a real
person's face re-lip-synced to the Hindi translation. It runs on a GPU (the T4
from [`KAGGLE.md`](KAGGLE.md)), offline. Every asset URL below is verified.

> The video used here has **no audio track** (silent B-roll), so we drive the
> translation from a separate English clip (a FLEURS sentence) via `--audio`. The
> face just needs to be a frontal talking head; Wav2Lip overwrites the mouth.

## 0. Prereqs (once per session)
GPU + Internet on. Then set up Vaani + deps exactly as in `KAGGLE.md`, plus video
deps, and prep the FLEURS clip we'll "speak":
```python
!git clone https://github.com/heroarnab2000/Vaani.git
%cd Vaani
!pip install -q faster-whisper "transformers>=4.57,<5" sentencepiece sacrebleu datasets coqui-tts speechbrain
!pip install -q imageio imageio-ffmpeg pillow
import os; os.environ["COQUI_TOS_AGREED"] = "1"
!python scripts/prep_fleurs.py --n 12          # gives us data/samples/fleurs_en_us_1938.wav
```

## 1. Get the real face video (verified URL)
```python
!curl -sL -o data/samples/pexels_src.mp4 \
  https://videos.pexels.com/video-files/8136210/8136210-hd_1080_1920_25fps.mp4
!ls -la data/samples/pexels_src.mp4     # ~5.7 MB, 12.8s, 1080x1920, a man talking to camera
```
*(If that URL ever 404s, grab a fresh one from https://www.pexels.com/video/a-man-talking-to-the-camera-8136210/ — the "Download" button, or upload your own clip.)*

## 2. Set up Wav2Lip + weights (verified mirrors — the usual blocker solved)
```python
!git clone https://github.com/Rudrabha/Wav2Lip
!curl -sL -o Wav2Lip/checkpoints/wav2lip_gan.pth \
  https://huggingface.co/spaces/manavisrani07/gradio-lipsync-wav2lip/resolve/main/checkpoints/wav2lip_gan.pth
!mkdir -p Wav2Lip/face_detection/detection/sfd
!curl -sL -o Wav2Lip/face_detection/detection/sfd/s3fd.pth \
  https://huggingface.co/spaces/manavisrani07/gradio-lipsync-wav2lip/resolve/main/face_detection/detection/sfd/s3fd.pth
# Wav2Lip is old: pin a compatible librosa and patch the removed np.float alias.
!pip install -q librosa==0.9.1 numba==0.58 opencv-python-headless
!sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g' Wav2Lip/audio.py Wav2Lip/inference.py Wav2Lip/hparams.py
```
> If Wav2Lip's deps still fight Kaggle's image, the actively-maintained
> **Easy-Wav2Lip** (https://github.com/anothermartz/Easy-Wav2Lip) auto-installs
> everything and its `install.py` fetches the checkpoints — a reliable fallback.

## 3. Run it — one command (translate + lip-sync)
```python
!python scripts/lipsync_demo.py data/samples/pexels_src.mp4 \
    --audio data/samples/fleurs_en_us_1938.wav \
    --config configs/gpu_lipsync.yaml \
    --out data/samples/result.mp4
```
This runs: load the man's frames + the English FLEURS clip → ASR→MT→TTS (Hindi in
a cloned voice, isochrony on) → **Wav2Lip re-syncs his mouth to the Hindi** → MP4.

## 4. Watch the result
```python
from IPython.display import Video
Video("data/samples/result.mp4", embed=True, width=360)
```

## How it maps to the code
- `scripts/lipsync_demo.py` — orchestrates it; `--audio` supplies English when the
  video is silent.
- `src/s2st/lipsync/wav2lip.py` — `Wav2LipStage` shells to `Wav2Lip/inference.py`
  (`configs/gpu_lipsync.yaml` sets `stages.lipsync: wav2lip` + the checkpoint path).
- `src/s2st/video/io.py` — frame read/write + audio mux.
- Swapping back to the dummy is one line: `stages.lipsync: dummy`.

## Honest limits
- **Offline only** — file in, file out. Real-time ≤5s (Phase 7) needs streaming first.
- **Voice** is still the Hindi-TTS weak spot (SECS ~0.31): lips will match, but the
  voice isn't fully convincing yet — tracked in [`VIDEO_ROADMAP.md`](VIDEO_ROADMAP.md) R6.
- **Frontal faces only** — Wav2Lip needs a roughly head-on talking face.
