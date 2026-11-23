"""The actual dub pipeline: mkv + srt + language -> one audio file.

v1 simplification (documented, not accidental): no diarization yet, so one
auto-selected reference clip clones a single voice for the entire file. See
plan.md's "Open questions in Doblarr" for what adding real diarization would
change later - that's a separate, bigger piece of work.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from . import config

_model = None


def get_model():
    """Load once per worker process - this is the expensive part."""
    global _model
    if _model is None:
        device = "xpu" if torch.xpu.is_available() else "cpu"
        print(f"[pipeline] loading ChatterboxMultilingualTTS on {device}", flush=True)
        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        if device != "cpu":
            model.ve.to(device)
            model.t3.to(device)
            model.s3gen.to(device)
            model.device = device
            if model.conds is not None:
                model.conds = model.conds.to(device)
        _model = model
    return _model


def _to_seconds(ts: str) -> float:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", content.strip())
    cues = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 2:
            continue
        m = re.match(r"(\d\d:\d\d:\d\d,\d+) --> (\d\d:\d\d:\d\d,\d+)", lines[1])
        if not m:
            continue
        start, end = _to_seconds(m.group(1)), _to_seconds(m.group(2))
        text = " ".join(lines[2:]).strip()
        if text:
            cues.append((start, end, text))
    return cues


def pick_reference_window(cues: list[tuple[float, float, str]]) -> tuple[float, float]:
    """Pick a clean, single-line cue to use as the voice-clone reference.

    Same heuristic Doblarr uses: not too short, not too long, closest to a
    target length, skipping likely intro/title cues at the very start.
    """
    candidates = [
        (start, end)
        for start, end, _ in cues
        if start >= config.REF_SKIP_SECONDS
        and config.REF_MIN_DURATION <= (end - start) <= config.REF_MAX_DURATION
    ]
    if not candidates:
        # fall back to the longest cue available, even outside preferred range
        candidates = [(s, e) for s, e, _ in cues] or [(0.0, 5.0)]
    candidates.sort(key=lambda se: abs((se[1] - se[0]) - config.REF_TARGET_DURATION))
    return candidates[0]


def extract_reference_clip(mkv_path: Path, start: float, end: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(mkv_path),
            "-map", "0:a:0",
            "-ss", str(start), "-to", str(end),
            "-ac", "1", "-ar", "16000",
            "-af", "loudnorm=I=-14",
            str(out_path),
        ],
        check=True, capture_output=True,
    )


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    )
    return float(out.strip())


def generate_dub_audio(mkv_path: Path, srt_path: Path, language: str, work_dir: Path) -> Path:
    """Full pipeline for one (mkv, srt, language). Returns path to the output wav."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cues = parse_srt(srt_path)
    if not cues:
        raise ValueError("no cues found in srt")

    ref_start, ref_end = pick_reference_window(cues)
    ref_path = work_dir / "ref.wav"
    extract_reference_clip(mkv_path, ref_start, ref_end, ref_path)
    print(f"[pipeline] reference clip: {ref_start:.1f}-{ref_end:.1f}s", flush=True)

    model = get_model()
    total_duration = _ffprobe_duration(mkv_path)

    clip_paths: list[tuple[float, Path]] = []
    for i, (start, end, text) in enumerate(cues):
        wav = model.generate(text, language_id=language, audio_prompt_path=str(ref_path))
        raw = work_dir / f"cue_{i:04d}_raw.wav"
        sf.write(raw, wav.squeeze().cpu().numpy(), model.sr)

        slot = max(end - start, 0.3)
        dur = _ffprobe_duration(raw)
        if dur > slot:
            atempo = min(dur / slot, config.MAX_ATEMPO)
            fitted = work_dir / f"cue_{i:04d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw), "-filter:a", f"atempo={atempo}", str(fitted)],
                check=True, capture_output=True,
            )
        else:
            fitted = raw
        clip_paths.append((start, fitted))
        if i % 25 == 0:
            print(f"[pipeline] cue {i}/{len(cues)}", flush=True)

    inputs = ["-f", "lavfi", "-t", str(total_duration), "-i", "anullsrc=r=24000:cl=mono"]
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

    out_path = work_dir / f"dub_{language}.wav"
    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[aout]", str(out_path)],
        check=True, capture_output=True,
    )
    return out_path
