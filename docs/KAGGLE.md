# Running Vaani on a free Kaggle GPU

The full cascade is ~5.5× slower than real time on CPU; a Kaggle **T4 (16 GB,
free, ~30 GPU-hrs/week)** fits all ~6.5 GB of models and cuts a 12-clip eval from
~18 min to a couple of minutes. This is the cheapest way to iterate before
committing to paid hardware.

## 1. New notebook + turn the GPU on
1. https://www.kaggle.com → **Create → New Notebook**.
2. Right sidebar → **Settings**:
   - **Accelerator** = `GPU T4 x2` (we use one GPU; two is fine).
   - **Internet** = `On` (needed to clone the repo and download models).

## 2. Get the code
```python
!git clone https://github.com/heroarnab2000/Vaani.git
%cd Vaani
```

## 3. Install deps — use Kaggle's GPU torch, add the rest
**Do NOT reinstall torch.** Kaggle already ships a CUDA build (your step-4 check
proves it works), and the repo's `torch==2.8.0` pin is a *Windows-laptop* choice
that doesn't exist on the cu121 wheel index (it tops out at 2.5.x). Just add the
project's other deps on top of Kaggle's torch:
```python
!pip install -q faster-whisper "transformers>=4.57,<5" sentencepiece sacrebleu \
    datasets coqui-tts speechbrain
```
> After this, re-check `torch.cuda.is_available()` is still `True`. If a dep
> downgraded torch to CPU, restart the kernel and pin an **available** CUDA build:
> `torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`.
> If pip reports a resolver conflict, restart the kernel and re-run (skip re-cloning).

## 4. Sanity-check the GPU is visible
```python
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```
Expect `CUDA: True Tesla T4`.

## 5. Prep the test data (first run downloads the FLEURS slice + models)
```python
import os; os.environ["COQUI_TOS_AGREED"] = "1"   # accept XTTS non-commercial license
!python scripts/prep_fleurs.py --n 12
```

## 6. Run — on the GPU config
```python
# metric table over the 12-clip slice (models download lazily on first call)
!python scripts/run_eval.py --config configs/gpu.yaml

# one clip end-to-end -> Hindi wav in the source voice
!python scripts/demo.py --config configs/gpu.yaml --id fleurs_1938
```
`configs/gpu.yaml` is `default.yaml` with every stage on `device: cuda`. To also
score voice preservation / naturalness, set `secs: true` / `utmos: true` in that
file (or edit it in the notebook).

## Notes & gotchas
- **First run downloads ~6.5 GB of models** (Whisper + NLLB + XTTS). Kaggle's
  internet is fast, so this is minutes, not the ~hour it is locally.
- **Storage is ephemeral** — the model cache is wiped when the session ends. To
  avoid re-downloading every session, save the HF cache (`~/.cache/huggingface`)
  as a **Kaggle Dataset** and attach it, or just accept the re-download (fast).
- **Sessions time out** (idle ~20 min, max ~12 hr) and are **not** for a live
  real-time server — Kaggle is for *development and evals*. The live ≤5s demo
  (Phase 7) needs a persistent GPU (rented cloud instance or owned card).
- **Expected speedup:** RTF should drop from ~5.5 toward ~1 (T4, pre-quantization);
  a 12-clip eval goes from ~18 min to ~2–3 min. Confirm with the printed RTF.
- The **dummy path still works** with zero GPU (`--config configs/dummy.yaml`) if
  you just want to check plumbing.
