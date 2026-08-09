# Results log

Measured numbers as real stages land. This is the payoff of the Phase 0
plumbing: results flow through the same harness unchanged. Every row is
reproducible via `scripts/prep_fleurs.py` + `scripts/run_eval.py`.

Test set: **FLEURS** `en_us` test, first **12** utterances (~107s of speech),
English→Hindi. Hindi references matched by FLEURS sentence id (11/12 available).
Hardware: RTX 3050 Ti laptop (4 GB), but all numbers below are **CPU INT8**.

## Phase 1 — cascade baseline (full cascade running end-to-end)

distil-large-v3 (ASR) → NLLB-200 distilled 600M (MT) → XTTS-v2 (voice-cloning
TTS), all **CPU**. English clip in → Hindi audio in the source speaker's voice.

| Stage | Model | Metric | Value (median / mean) | Notes |
|---|---|---|---|---|
| ASR | distil-large-v3 (CT2 INT8) | WER | **0.058 / 0.060** | English; normalized text |
| Translation | NLLB-200 distilled 600M | spBLEU | **31.21** | flores200 tokenizer; 11 refs |
| Translation | NLLB-200 distilled 600M | COMET | _optional_ | wired; enable `eval.comet` (~2.3 GB) |
| TTS | XTTS-v2 | duration_deviation | **0.477 / 0.962** | isochrony "before" (Phase 2 → ~0.10-0.15) |
| TTS | XTTS-v2 | SECS (ECAPA) | **0.309 / 0.279** | 12-clip En→Hi; low — mostly cross-lingual, see note |
| TTS | XTTS-v2 | UTMOS (SpeechMOS) | **2.644 / 2.659** | 12-clip En→Hi; mediocre in Hindi, see note |
| System | full cascade | RTF | **5.57 / 6.65** | ~45s/clip on CPU; Phase-4 "before" |

Notes:
- The **duration_deviation 0.48 median** is the deliberate ugly baseline the
  Phase 2 isochrony work is measured against; RTF 5.57 is the Phase 4 "before".
- `fleurs_1776` is a duration outlier (4.4) that skews the mean — use the median.
- distil-large-v3 over large-v3 (English source, ~equal WER, half the download);
  swap to `large-v3` in `configs/default.yaml` for a multilingual run.
- Demo artifact: `python scripts/demo.py --id <id>` writes the translated wav.
- **SECS / UTMOS measured** (full 12-clip En→Hi, isochrony off, models cached):
  SECS median **0.309** / mean 0.279; UTMOS median **2.644** / mean 2.659. Both
  sit well below a strong TTS (SECS ~0.7-0.9, UTMOS ~3.5+), and the low SECS is
  *uniform* (0.22-0.49), not outlier-driven — so voice preservation is genuinely
  weak in the Hindi direction. Two clips are speaker-mismatch failures:
  `fleurs_1776` (SECS **-0.006**, worst WER 0.20) and `fleurs_1876` (SECS
  **0.002**, lowest UTMOS 1.48, dur-dev 1.55).
- **Cross-lingual control** (En→En resynthesis — XTTS re-clones each speaker into
  *English* from the same reference, `ref_src` text): SECS median **0.546** /
  mean 0.501, UTMOS median **3.838** / mean 3.794 — far above the Hindi numbers.
  **Conclusion: the loss is mostly the Hindi direction, not broken cloning.** The
  two Hindi "failures" recover in English (1876: 0.002→0.578, 1776: -0.006→0.404)
  — they were bad *Hindi* syntheses, not broken ones. Caveats: (a) even En→En
  SECS (0.55) is below the 0.7-0.9 ideal → a residual cloning/methodology gap
  (reference length, leading silence, or ECAPA over full clips); (b) two speakers
  (1972, 1914) clone poorly even in English (SECS ~0.22-0.31).
- **Implication:** improving voice preservation means improving the **Hindi TTS**
  — longer/cleaner speaker refs or XTTS speaker latents, an XTTS Hindi fine-tune,
  or a stronger Hindi TTS for the Hi direction — plus the 150-char / number fixes
  below. This matters extra for the video goal ([VIDEO_ROADMAP.md](VIDEO_ROADMAP.md)
  risk R6): a weak voice under re-synced lips looks like the person but doesn't
  *sound* like them. (Reproduce: `run_eval.py` with `eval.secs/utmos: true`; the
  En→En control drives the TTS stage directly with `tgt_lang='en'`.)

Known XTTS-v2 limitations (tracked, not yet fixed):
- 150-char/sentence cap can truncate a few long Hindi sentences.
- Hindi number verbalization: num2words lacks `hi`, so digits are mapped to
  Devanagari numerals (not spoken as words yet) to avoid a crash.

## Phase 2 — isochrony (on/off ablation)

Two levers behind the `isochrony.enabled` switch:
1. **Duration-aware MT decoding** — NLLB generates candidates at several length
   penalties; we keep the one whose predicted spoken length (chars/sec prior)
   best fits the source-speech duration.
2. **TTS rate control** — phase-vocoder time-stretch (pitch-preserving) fits the
   XTTS output to the budget, clamped to a 1.5× factor.

Same models / test set as Phase 1; only `isochrony.enabled` flips. Reproduce:
`python scripts/ablation.py`.

| metric | OFF (baseline) | ON (isochrony) | goal |
|---|---|---|---|
| duration_deviation (median) | 0.479 | **0.000** | ↓ |
| duration_deviation (mean) | 1.057 | **0.205** | ↓ |
| spBLEU | 31.21 | 30.31 | hold |
| WER (median) | 0.058 | 0.058 | unchanged |
| RTF (median) | ~5.2 | 8.56 | ↑ cost (Phase 4) |

Reading this honestly:
- Time-stretch can fit duration almost exactly, so dur-dev → 0 within the 1.5×
  clamp is **expected** — 8/12 clips hit 0. The real engineering value is keeping
  the stretch factor small (so fewer artifacts), which is what the MT reranking
  buys, and doing it **without hurting translation quality** (spBLEU 31.2 → 30.3,
  WER unchanged).
- 3 long-Hindi clips (1950, 1914, 1806) keep residual dev (0.35–1.31): even the
  shortest candidate, stretched to the 1.5× clamp, can't fully fit. Raising the
  clamp or stronger length control would help — at a naturalness cost.
- RTF rises because reranking runs NLLB 3×. That is a Phase-4 concern, not a
  Phase-2 one.
- Naturalness of the stretched audio (UTMOS) and the realized stretch factor are
  the right follow-up metrics to quantify the quality cost — currently deferred.
- **XTTS is stochastic** (sampling decoder), so output length and hence dur-dev
  vary run-to-run; the numbers above are a single 12-clip run. Small sets are
  noisy (a 2-clip dev run swung OFF 0.07 / ON 0.19) — trust the 12-clip figures,
  and averaging over seeds is a deferred improvement.
