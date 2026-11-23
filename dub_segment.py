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

    # Build the dub track: the ORIGINAL audio (music/SFX/dialogue) at reduced
    # volume as a "bed", sidechain-ducked under the new dialogue bus - not
    # silence. Ported from doblarr/stages/mix.py (the real Doblarr source):
    # a first naive version (silence base + default amix) came out "way too
    # quiet with no sound effects audible" - amix defaults to normalize=1
    # (divides volume by input count), and silence obviously has no bed to
    # preserve. Doblarr's own no-Demucs fallback is exactly this: original
    # audio at fallback_volume=0.2, sidechain-ducked by the dialogue.
    window_dur = args.end - args.start
    bed_fallback_volume = 0.2
    duck_threshold, duck_ratio, duck_attack_ms, duck_release_ms = 0.02, 12.0, 20, 250

    inputs = ["-ss", str(args.start), "-to", str(args.end), "-i", args.mkv]  # [0:a] = original -> bed
    filter_parts = [f"[0:a]volume={bed_fallback_volume:g},aformat=channel_layouts=stereo[bed]"]
    for i, (offset_s, path) in enumerate(clip_paths):
        inputs += ["-i", str(path)]
        delay_ms = max(int(offset_s * 1000), 0)
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms},aformat=channel_layouts=stereo[c{i}]")

    if not clip_paths:
        filter_parts.append("[bed]acopy[out]")
    else:
        clip_refs = "".join(f"[c{i}]" for i in range(len(clip_paths)))
        if len(clip_paths) == 1:
            filter_parts.append(f"[c0]apad=whole_dur={window_dur:g},asplit=2[dlg][key]")
        else:
            filter_parts.append(
                f"{clip_refs}amix=inputs={len(clip_paths)}:normalize=0,"
                f"apad=whole_dur={window_dur:g},asplit=2[dlg][key]"
            )
        filter_parts.append(
            f"[bed][key]sidechaincompress=threshold={duck_threshold:g}:ratio={duck_ratio:g}"
            f":attack={duck_attack_ms:g}:release={duck_release_ms:g}:makeup=1[ducked]"
        )
        filter_parts.append("[ducked][dlg]amix=inputs=2:normalize=0[master]")
        filter_parts.append("[master]alimiter=limit=0.95:level=false:latency=true[out]")

    dub_track = work / "dub_track.wav"
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filter_parts), "-map", "[out]", str(dub_track)],
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
