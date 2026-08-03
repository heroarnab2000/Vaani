#!/usr/bin/env python3
"""Run one clip through the full cascade and save the translated audio.

The listenable artifact behind the project's demo: English in -> Hindi out in
the source speaker's voice. Prints the ASR transcript and the translation.

Usage:
    python scripts/demo.py                  # first item of the eval manifest
    python scripts/demo.py --id fleurs_1938 # a specific manifest item
    python scripts/demo.py path/to/clip.wav # an arbitrary English wav
"""
import argparse
import json
import sys
from pathlib import Path

# print Hindi (Devanagari) safely regardless of the console code page
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import soundfile as sf

from s2st.audio import load_wav
from s2st.config import load_config, resolve_path
from s2st.factory import build_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="?", help="English wav (default: first manifest item)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--id", default=None, help="manifest item id to use")
    ap.add_argument("--out-dir", default="data/samples")
    ap.add_argument(
        "--isochrony", choices=["on", "off"], default=None,
        help="override isochrony.enabled (A/B the duration matching)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.isochrony is not None:
        cfg.setdefault("isochrony", {})["enabled"] = args.isochrony == "on"
    sr = cfg.get("audio", {}).get("sample_rate", 16000)

    ref_text = None
    if args.audio:
        wav_path = Path(args.audio)
    else:
        manifest = resolve_path(cfg.get("eval", {}).get("manifest"))
        items = json.loads(manifest.read_text(encoding="utf-8"))
        item = items[0]
        if args.id:
            item = next((x for x in items if x.get("id") == args.id), items[0])
        wav_path = resolve_path(item["audio"])
        ref_text = item.get("ref_tgt")
        print(f"item: {item.get('id')}")
        print(f"ref_src: {item.get('ref_src')}")

    audio = load_wav(wav_path, sr)
    if audio is None:
        raise SystemExit(f"cannot read audio: {wav_path}")

    pipe = build_pipeline(cfg)
    out = pipe.run(audio, sr, tgt_lang=cfg.get("language", {}).get("tgt", "hi"))

    iso_on = bool(cfg.get("isochrony", {}).get("enabled"))
    src_dur = len(audio) / sr
    print(f"ASR : {out.asr.text}")
    print(f"MT  : {out.translation.text}")
    if ref_text:
        print(f"ref : {ref_text}")
    print(f"isochrony={'on' if iso_on else 'off'}  "
          f"source={src_dur:.1f}s  target={out.translation.target_speech_duration:.1f}s  "
          f"output={out.tts.realized_duration:.1f}s")
    print(f"timings(s): {{'asr': {out.timings['asr']:.2f}, "
          f"'mt': {out.timings['translation']:.2f}, 'tts': {out.timings['tts']:.2f}}}")

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "iso-on" if iso_on else "iso-off"
    out_wav = out_dir / f"demo_{wav_path.stem}_hi_{tag}.wav"
    sf.write(str(out_wav), out.tts.audio, out.tts.sample_rate)
    print(f"wrote translated audio -> {out_wav} "
          f"({out.tts.realized_duration:.1f}s @ {out.tts.sample_rate} Hz)")


if __name__ == "__main__":
    main()
