#!/usr/bin/env python3
"""Pull a small FLEURS slice into wav files + an eval manifest.

FLEURS (google/fleurs) is sentence-parallel across languages by integer `id`,
so the English audio carries an English reference transcript (for WER) and we
can attach the matching Hindi transcript (for BLEU/COMET) by id.

Streamed, so only the rows we use are downloaded -- no full-dataset pull. We
cast the audio column to *no-decode* and write the raw wav bytes ourselves,
which avoids datasets 4.x's torchcodec audio-decoding dependency.

Usage:
    python scripts/prep_fleurs.py --n 15
    python scripts/prep_fleurs.py --n 15 --no-hindi
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


def _load_stream(config: str, split: str):
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", config, split=split, streaming=True)
    # don't let datasets decode audio (it would require torchcodec); we get
    # {"bytes", "path"} and handle the wav ourselves.
    return ds.cast_column("audio", Audio(decode=False))


def _write_wav(raw: dict, dest: Path) -> float:
    """Write a FLEURS audio entry to dest; return duration in seconds."""
    data = raw.get("bytes")
    if data:
        dest.write_bytes(data)
    elif raw.get("path"):
        shutil.copy(raw["path"], dest)
    else:
        raise ValueError("audio entry has neither bytes nor path")
    info = sf.info(str(dest))
    return info.frames / info.samplerate


def fetch_hindi_refs(needed_ids: set, hi_config: str, split: str, scan_cap: int) -> dict:
    """Stream the Hindi split and collect transcripts for the needed ids."""
    refs: dict = {}
    try:
        ds = _load_stream(hi_config, split)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load Hindi refs ({hi_config}): {e}")
        return refs
    for i, ex in enumerate(ds):
        if i >= scan_cap or len(refs) >= len(needed_ids):
            break
        if ex["id"] in needed_ids:
            refs[ex["id"]] = ex["transcription"]
    return refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=15, help="number of English samples")
    ap.add_argument("--en-config", default="en_us")
    ap.add_argument("--hi-config", default="hi_in")
    ap.add_argument("--split", default="test")
    ap.add_argument("--no-hindi", action="store_true", help="skip Hindi reference fetch")
    ap.add_argument("--scan-cap", type=int, default=5000, help="max Hindi rows to scan")
    ap.add_argument(
        "--out",
        default=str(SAMPLES_DIR / "fleurs_manifest.json"),
        help="output manifest path",
    )
    args = ap.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"streaming google/fleurs {args.en_config} [{args.split}], taking {args.n}...")
    en = _load_stream(args.en_config, args.split)

    items = []
    for ex in en:
        if len(items) >= args.n:
            break
        wav_path = SAMPLES_DIR / f"fleurs_{args.en_config}_{ex['id']}.wav"
        dur = _write_wav(ex["audio"], wav_path)
        items.append(
            {
                "id": f"fleurs_{ex['id']}",
                "_fleurs_id": ex["id"],
                "audio": str(wav_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "ref_src": ex["transcription"],
                "ref_tgt": "",
                "tgt_lang": "hi",
            }
        )
        print(f"  + {wav_path.name}  ({dur:.1f}s)")

    if not args.no_hindi and items:
        needed = {it["_fleurs_id"] for it in items}
        print(f"fetching Hindi refs for {len(needed)} ids ({args.hi_config})...")
        hi = fetch_hindi_refs(needed, args.hi_config, args.split, args.scan_cap)
        for it in items:
            it["ref_tgt"] = hi.get(it["_fleurs_id"], "")
        print(f"  matched {sum(1 for it in items if it['ref_tgt'])}/{len(items)} Hindi refs")

    for it in items:
        it.pop("_fleurs_id", None)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()
