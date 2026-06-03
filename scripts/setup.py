#!/usr/bin/env python3
"""One-shot environment bootstrap for the Vaani speech-to-speech cascade.

On a fresh clone this:
  1. creates a local .venv,
  2. installs the pinned dependency stack (CPU torch + the rest),
  3. pre-downloads the three models named in configs/default.yaml
     (ASR / MT / TTS) into the Hugging Face cache,
  4. fetches a small FLEURS test slice (wav + manifest).

Models are NOT committed to git; this script pulls them on demand. Safe to
re-run -- existing venv is reused and HF downloads resume from cache.

    python scripts/setup.py              # full setup (CPU)
    python scripts/setup.py --no-data    # skip the FLEURS test slice
    python scripts/setup.py --n 6        # smaller FLEURS slice
    python scripts/setup.py --no-models  # deps + data only (skip model pull)

Run it with any Python 3.10-3.12; it only uses the standard library until it
shells out to the freshly created .venv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV = REPO / ".venv"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# kept in sync with requirements.txt (torch <2.9 avoids torchcodec/FFmpeg on Win)
TORCH_PIN = ["torch==2.8.0", "torchaudio==2.8.0"]


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, env=None) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def py_c(py: str, code: str, env=None) -> None:
    run([py, "-c", code], env=env)


def step(msg: str) -> None:
    print(f"\n{'=' * 4} {msg} {'=' * 4}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-data", action="store_true", help="skip the FLEURS test slice")
    ap.add_argument("--no-models", action="store_true", help="skip model downloads")
    ap.add_argument("--n", type=int, default=12, help="FLEURS sample count")
    args = ap.parse_args()

    # 1. venv -----------------------------------------------------------------
    step("1/5  virtual environment")
    if venv_python().exists():
        print(f".venv already present -> {venv_python()}")
    else:
        run([sys.executable, "-m", "venv", str(VENV)])
    py = str(venv_python())

    # 2. dependencies ---------------------------------------------------------
    step("2/5  dependencies (this can take a while on first run)")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", *TORCH_PIN, "--index-url", TORCH_CPU_INDEX])
    run([py, "-m", "pip", "install", "-r", str(REPO / "requirements.txt")])

    # 3. model names from the config (single source of truth) -----------------
    step("3/5  models named in configs/default.yaml")
    cfg = REPO / "configs" / "default.yaml"
    out = subprocess.run(
        [py, "-c",
         f"import yaml,json;print(json.dumps(yaml.safe_load(open(r'{cfg}',encoding='utf-8')).get('models',{{}})))"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    models = json.loads(out)
    asr = models.get("asr", {}).get("name", "distil-large-v3")
    mt = models.get("translation", {}).get("name", "facebook/nllb-200-distilled-600M")
    tts = models.get("tts", {}).get("name", "tts_models/multilingual/multi-dataset/xtts_v2")
    print(f"ASR = {asr}\nMT  = {mt}\nTTS = {tts}")

    env = {
        **os.environ,
        "COQUI_TOS_AGREED": "1",            # accept XTTS non-commercial license
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "PYTHONUTF8": "1",
    }

    # 4. download models (each in its own process so RAM is released) ----------
    step("4/5  download models (HF cache; ~6 GB total, resumes if interrupted)")
    if args.no_models:
        print("skipped (--no-models)")
    else:
        print("-> ASR  (faster-whisper / CTranslate2)")
        py_c(py, f"from faster_whisper import WhisperModel; "
                 f"WhisperModel('{asr}', device='cpu', compute_type='int8'); print('ASR ok')", env)
        print("-> MT   (transformers / NLLB-200)")
        py_c(py, f"from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; "
                 f"AutoTokenizer.from_pretrained('{mt}'); "
                 f"AutoModelForSeq2SeqLM.from_pretrained('{mt}'); print('MT ok')", env)
        print("-> TTS  (coqui-tts / XTTS-v2)")
        py_c(py, f"from TTS.api import TTS; TTS('{tts}'); print('TTS ok')", env)

    # 5. test data ------------------------------------------------------------
    step("5/5  FLEURS test slice")
    manifest = REPO / "data" / "samples" / "fleurs_manifest.json"
    if args.no_data:
        print("skipped (--no-data)")
    elif manifest.exists():
        print(f"manifest already present -> {manifest} (delete to re-fetch)")
    else:
        run([py, str(REPO / "scripts" / "prep_fleurs.py"), "--n", str(args.n)], env=env)

    sep = "\\" if os.name == "nt" else "/"
    print("\nSetup complete. Try:")
    print(f"  {py} scripts{sep}demo.py --id fleurs_1938")
    print(f"  {py} scripts{sep}run_eval.py")


if __name__ == "__main__":
    main()
