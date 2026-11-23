"""Dub a time window of an episode into a target language using cross-lingual
voice cloning, placing each line at its subtitle timestamp.

Usage (inside the jelly-dub container):
  python dub_segment.py --mkv futurama-s09e02.mkv --srt pl.srt --ref ref_en_clip.wav \
      --lang pl --start 720 --end 870 --out dubbed_segment.mkv
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def parse_srt(path):
    content = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", content.strip())
    cues = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 2:
            continue
        m = re.match(r"(\d\d:\d\d:\d\d,\d+) --> (\d\d:\d\d:\d\d,\d+)", lines[1])
        if not m:
            continue
        start = to_seconds(m.group(1))
        end = to_seconds(m.group(2))
        text = " ".join(lines[2:]).strip()
        if text:
            cues.append((start, end, text))
    return cues


def to_seconds(ts):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mkv", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--ref", required=True, help="reference voice clip (wav)")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/_dub_tmp")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    cues = [c for c in parse_srt(args.srt) if args.start <= c[0] <= args.end]
    print(f"{len(cues)} cues in [{args.start}, {args.end}]", flush=True)

    device = "xpu" if torch.xpu.is_available() else "cpu"
    print(f"using device: {device}", flush=True)

    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    if device != "cpu":
        model.ve.to(device)
        model.t3.to(device)
        model.s3gen.to(device)
        model.device = device
        if model.conds is not None:
            model.conds = model.conds.to(device)

    clip_paths = []
    for i, (start, end, text) in enumerate(cues):
        wav = model.generate(text, language_id=args.lang, audio_prompt_path=args.ref)
        raw = work / f"cue_{i:03d}_raw.wav"
        sf.write(raw, wav.squeeze().cpu().numpy(), model.sr)

        slot = max(end - start, 0.3)
        dur = float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(raw)]
            ).strip()
        )
        fitted = work / f"cue_{i:03d}.wav"
        if dur > slot:
            atempo = min(dur / slot, 2.0)  # cap stretch so it doesn't get silly-fast
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw), "-filter:a", f"atempo={atempo}", str(fitted)],
                check=True, capture_output=True,
            )
        else:
            fitted = raw
        clip_paths.append((start - args.start, fitted))
        print(f"cue {i}: {text[:40]!r} -> {fitted.name}", flush=True)

    # Build the dub track: silence base + each clip delayed to its offset.
    window_dur = args.end - args.start
    inputs = ["-f", "lavfi", "-t", str(window_dur), "-i", "anullsrc=r=24000:cl=mono"]
    filter_parts = []
    mix_inputs = ["[0:a]"]
    for i, (offset_s, path) in enumerate(clip_paths):
        inputs += ["-i", str(path)]
        delay_ms = max(int(offset_s * 1000), 0)
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[d{i}]")
        mix_inputs.append(f"[d{i}]")
    filter_complex = ";".join(filter_parts)
    if filter_complex:
        filter_complex += f";{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:dropout_transition=0[aout]"
    else:
        filter_complex = "[0:a]acopy[aout]"
    dub_track = work / "dub_track.wav"
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[aout]", str(dub_track)],
        check=True,
    )

    # Mux: trim the video+original audio to the window, add the dub as a second audio track.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(args.start), "-to", str(args.end), "-i", args.mkv,
            "-i", str(dub_track),
            "-map", "0:v", "-map", "0:a", "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac",
            "-metadata:s:a:1", f"language={args.lang}",
            "-metadata:s:a:1", f"title={args.lang.upper()} AI dub (test)",
            args.out,
        ],
        check=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
