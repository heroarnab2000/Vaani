# Real-Time On-Device Speech-to-Speech Translation (En ↔ Hi)

Streaming ASR → isochrony-controlled translation → voice-preserving TTS, built to run faster than real time on edge hardware, with a full evaluation harness and a cascade-vs-direct ablation.

> This is a portfolio project aimed at AI/ML engineering roles (edge inference, model optimization). The design philosophy: **modify and optimize real models, prove every step with a before/after number** — not API plumbing. Full rationale, learning roadmap, and resource links are in [`docs/PROJECT_DOCS.md`](docs/PROJECT_DOCS.md).

## Status: Phase 0 complete

The whole pipeline runs end-to-end with **dummy stages**, and the eval harness prints a metric table. Real models get swapped in stage by stage without touching the orchestrator or harness.

## Quickstart

```bash
# no heavy deps needed for Phase 0
python tests/test_pipeline.py     # 6 tests pass
python scripts/run_eval.py        # prints the metric table
```

Example output:

```
=== S2ST evaluation report ===
items: 2

id              WER  dur_dev      RTF   lat(s)
----------------------------------------------
demo01        0.000    0.095    0.001    0.004
...
```

The numbers are meaningless with dummy stages — that's the point. Phase 0 proves the *plumbing* (every metric computes, aggregates, prints) so that real results later flow through unchanged.

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
  asr/              # DummyASR -> FasterWhisperASR (Phase 1)
  translation/      # DummyTranslation -> NLLB/Seamless (+isochrony Phase 2)
  tts/              # DummyTTS -> XTTSv2 (Phase 1)
  pipeline/         # orchestrator (model-agnostic, times each stage)
  eval/             # metrics + harness  (Phase 0 deliverable)
scripts/run_eval.py # entry point
configs/default.yaml
tests/
docs/PROJECT_DOCS.md
```

## How to extend (the core pattern)

Each real model implements one interface from `interfaces.py` and is selected in `configs/default.yaml`. To add real ASR:

1. Create `src/s2st/asr/faster_whisper.py` implementing `ASRStage.transcribe`.
2. Export it from `src/s2st/asr/__init__.py`.
3. Flip `stages.asr: faster_whisper` in the config.

Nothing else changes — the orchestrator and harness only know the interfaces.

## Roadmap

| Phase | Goal | Target metric |
|---|---|---|
| 0 ✅ | Scaffold + eval harness | plumbing works end-to-end |
| 1 | Cascade baseline (real models) | record baseline WER/BLEU/SECS/UTMOS/RTF |
| 2 | Isochrony (the differentiator) | median duration deviation ≈ 10–15% |
| 3 | Streaming | first-audio latency < 1–2s, RTF < 1 |
| 4 | Quantization + profiling | 2–4× latency gain, quality within ~2–5% |
| 5 | Edge deployment | RTF < 1 on device, no cloud |
| 6 | Cascade-vs-direct ablation | data-backed recommendation |
| 7 | (optional) lip-sync + personalization | LSE-C/D reported |

See [`docs/PROJECT_DOCS.md`](docs/PROJECT_DOCS.md) for the full plan, the learning roadmap (tiers), and the verified resource library.

## License / models

Verify individual model licenses before any commercial use (XTTS-v2 and some checkpoints are non-commercial). Fine for a portfolio — just label accurately.
