"""Single-slot dispatch loop: claims one queued job at a time and runs the
(blocking, GPU-bound) pipeline in a thread so the event loop stays responsive
for status polling. Mirrors gateway/app/dispatcher.py's shape.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from . import config, db, pipeline

# The realistic failure modes of one job: a missing/corrupt input file, a bad
# srt (ValueError in pipeline.parse_srt), an ffmpeg step failing
# (CalledProcessError), or a torch/runtime error from the model itself. Any of
# these must fail just that one job, not take down the dispatcher loop -
# listed explicitly rather than a bare `except Exception` so a genuinely
# unexpected error (e.g. a programming bug) still surfaces instead of being
# silently swallowed as a "failed job".
JOB_FAILURE_TYPES = (OSError, ValueError, RuntimeError, subprocess.CalledProcessError)

_running = True


def stop() -> None:
    global _running
    _running = False


def _job_dir(job_id: str) -> Path:
    return Path(config.WORK_DIR) / job_id.replace(":", "_")


def _run_job(job: db.Job) -> None:
    job_dir = _job_dir(job.id)
    mkv_path = Path(config.WORK_DIR) / job.mkv_hash / "input.mkv"
    srt_path = job_dir / "input.srt"
    try:
        out_path = pipeline.generate_dub_audio(
            mkv_path, srt_path, job.language, job_dir,
            on_progress=lambda current, total: db.set_progress(job.id, current, total),
        )
        db.set_status(job.id, "done", output_path=str(out_path))
        print(f"[dispatcher] job {job.id} done -> {out_path}", flush=True)
    except JOB_FAILURE_TYPES as e:
        db.set_status(job.id, "failed", error=repr(e))
        print(f"[dispatcher] job {job.id} failed: {e!r}", flush=True)


async def run_forever() -> None:
    while _running:
        job = db.claim_next_queued_job()
        if job is None:
            await asyncio.sleep(config.DISPATCH_POLL_INTERVAL)
            continue
        print(f"[dispatcher] claimed {job.id}", flush=True)
        await asyncio.to_thread(_run_job, job)
