# Results log

Measured numbers as real stages land. This is the payoff of the Phase 0
plumbing: results flow through the same harness unchanged. Every row is
reproducible via `scripts/prep_fleurs.py` + `scripts/run_eval.py`.

Test set: **FLEURS** `en_us` test, first **12** utterances (~107s of speech),
English→Hindi. Hindi references matched by FLEURS sentence id (11/12 available).
Hardware: RTX 3050 Ti laptop (4 GB), but all numbers below are **CPU INT8**.

## Phase 1 — cascade baseline (in progress)

| Stage | Model | Metric | Value | Notes |
|---|---|---|---|---|
| ASR | distil-large-v3 (CT2 INT8, CPU) | WER | **0.058 median / 0.060 mean** | English; normalized text |
| ASR | distil-large-v3 (CT2 INT8, CPU) | RTF | **0.93 median / 1.19 mean** | ~9.4s/clip fixed encoder cost dominates |
| Translation | NLLB-200 distilled 600M | spBLEU / COMET | _pending_ | stage coded; not yet measured |
| TTS | XTTS-v2 | SECS / UTMOS / dur-dev | _pending_ | stage not yet built |

Notes:
- distil-large-v3 chosen over large-v3 for the baseline (English-only source,
  ~equal WER on English, half the download). Swap to `large-v3` in
  `configs/default.yaml` for a multilingual run.
- RTF > target is expected here — optimization is Phase 4. This row is the
  *before* number the Phase 4 deltas will be measured against.
- `duration_deviation` is meaningless until real TTS lands (dummy TTS still in
  the cascade), so it is omitted above.
