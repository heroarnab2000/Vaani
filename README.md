# Real-Time On-Device Speech-to-Speech Translation (En ↔ Hi)

Streaming ASR → isochrony-controlled translation → voice-preserving TTS, built to run faster than real time on edge hardware, with a full evaluation harness and a cascade-vs-direct ablation.

> This is a portfolio project aimed at AI/ML engineering roles (edge inference, model optimization). The design philosophy: **modify and optimize real models, prove every step with a before/after number** — not API plumbing. Full rationale, learning roadmap, and resource links are in [`docs/PROJECT_DOCS.md`](docs/PROJECT_DOCS.md).

## Status: Phase 2 — isochrony (duration-aware MT + TTS rate control)

distil-large-v3 (ASR) → NLLB-200 (MT) → XTTS-v2 (voice-cloning TTS) runs end to
end on CPU: **English clip in → Hindi audio in the source speaker's voice**, now
**length-matched to the source**. FLEURS en→hi, 12 clips:

| Metric | Phase 1 baseline | + Isochrony | Stage |
|---|---|---|---|
| WER (median) | 0.058 | 0.058 | ASR (distil-large-v3, CT2 INT8) |
| spBLEU | 31.21 | 30.31 | MT (NLLB-200 distilled 600M) |
| duration deviation (median / mean) | 0.48 / 1.06 | **0.00 / 0.21** | isochrony (MT rerank + TTS rate control) |
| RTF (median) | ~5.2 | 8.6 | full CPU cascade — Phase 4 "before" |

Reproduce the ablation: `python scripts/ablation.py`. Full table + honest
caveats in [`docs/RESULTS.md`](docs/RESULTS.md) (dur-dev → 0 is partly "by
construction" of rate control; the win is holding BLEU while doing it). SECS /
UTMOS / COMET still wired-but-unmeasured. Stages + isochrony toggle live in
[`configs/default.yaml`](configs/default.yaml); the orchestrator and harness
only ever touch the abstract interfaces.

## Quickstart

**One command** — creates `.venv`, installs the pinned stack, and downloads all
models + a FLEURS test slice (~6 GB of models on first run; resumes from cache):

```bash
python scripts/setup.py
```

Then, using the venv's Python (`./.venv/Scripts/python.exe` on Windows,
`./.venv/bin/python` elsewhere):

```bash
.venv/Scripts/python.exe scripts/run_eval.py            # metric table over the test slice
.venv/Scripts/python.exe scripts/demo.py --id fleurs_1938   # English in -> Hindi out, same voice
.venv/Scripts/python.exe tests/test_pipeline.py         # 8 plumbing tests (run on bare numpy)
```

<details><summary>Manual setup (if you'd rather not use <code>setup.py</code>)</summary>

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (source .venv/bin/activate on *nix)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/prep_fleurs.py --n 12   # models download lazily on first run
```
</details>

Stages are chosen in [`configs/default.yaml`](configs/default.yaml)
(`stages.asr/translation/tts`); set them all to `dummy` to run the original
zero-dependency Phase-0 pipeline.

## Architecture

```
mic ─► VAD/chunk ─► ASR (Whisper) ─► duration estimate ─► Translation (isochrony) ─► TTS (voice clone) ─► out
                                                                                          └─ (optional) lip-sync
```

Cascade by design (per-stage control, production-standard). A direct end-to-end model (SeamlessM4T) is built later *only* as an ablation baseline.

## Project structure

```
src/s2st/
  types.py          # dataclasses passed between stages
  interfaces.py     # ASRStage / TranslationStage / TTSStage ABCs
  config.py         # load configs/default.yaml
  factory.py        # config string -> concrete stage (lazy heavy imports)
  audio.py          # mono mix + linear resample + wav load (no heavy deps)
  asr/              # DummyASR | FasterWhisperASR (faster-whisper, CT2)
  translation/      # DummyTranslation | NLLBTranslation (NLLB-200)
  tts/              # DummyTTS | XTTSv2TTS (voice cloning; isochrony in Phase 2)
  pipeline/         # orchestrator (model-agnostic, times each stage)
  eval/             # metrics (WER/spBLEU/COMET/dur-dev/RTF) + harness
scripts/
  setup.py          # one-shot bootstrap: venv + deps + models + test data
  run_eval.py       # build pipeline from config -> metric table
  prep_fleurs.py    # stream a FLEURS slice -> wav + manifest (en/hi refs)
  demo.py           # one clip end-to-end -> saved translated wav
configs/default.yaml
docs/PROJECT_DOCS.md  docs/RESULTS.md   # plan + measured results log
tests/
```

## How to extend (the core pattern)

Each real model implements one interface from `interfaces.py` and is selected in `configs/default.yaml`. To add real ASR:

1. Create `src/s2st/asr/faster_whisper.py` implementing `ASRStage.transcribe`.
2. Add a branch in `src/s2st/factory.py` (`build_asr`) with a **lazy** import so
   the dummy path stays dependency-free.
3. Flip `stages.asr: faster_whisper` in the config.

Nothing else changes — the orchestrator and harness only know the interfaces.

## Roadmap

| Phase | Goal | Target metric |
|---|---|---|
| 0 ✅ | Scaffold + eval harness | plumbing works end-to-end |
| 1 ✅ (core) | Cascade baseline (real models) | WER 0.058, spBLEU 31.2, dur-dev 0.48, RTF 5.6 — SECS/UTMOS/COMET pending |
| 2 ✅ | Isochrony (duration-aware MT + TTS rate control) | dur-dev median 0.48 → 0.00, mean 1.06 → 0.21; spBLEU held (31.2 → 30.3), WER unchanged |
| 3 | Streaming | first-audio latency < 1–2s, RTF < 1 |
| 4 | Quantization + profiling | 2–4× latency gain, quality within ~2–5% |
| 5 | Edge deployment | RTF < 1 on device, no cloud |
| 6 | Cascade-vs-direct ablation | data-backed recommendation |
| 7 | (optional) lip-sync + personalization | LSE-C/D reported |

See [`docs/PROJECT_DOCS.md`](docs/PROJECT_DOCS.md) for the full plan, the learning roadmap (tiers), and the verified resource library.

## License / models

Verify individual model licenses before any commercial use (XTTS-v2 and some checkpoints are non-commercial). Fine for a portfolio — just label accurately.
