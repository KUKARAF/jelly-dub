"""Environment-driven settings for the dub worker."""

from __future__ import annotations

import os

WORK_DIR = os.getenv("WORKER_WORK_DIR", "/work/jobs")
DB_PATH = os.getenv("WORKER_DB_PATH", "/work/data/worker.sqlite3")

# Reference clip auto-selection: pick a subtitle cue in this duration range
# (seconds), preferring the one closest to the target length, as the voice
# clone reference - same heuristic Doblarr itself uses (see plan.md), just
# without diarization: one voice for the whole file is a known v1
# simplification, not an oversight.
REF_MIN_DURATION = float(os.getenv("WORKER_REF_MIN_DURATION", "3.0"))
REF_MAX_DURATION = float(os.getenv("WORKER_REF_MAX_DURATION", "12.0"))
REF_TARGET_DURATION = float(os.getenv("WORKER_REF_TARGET_DURATION", "8.0"))
REF_SKIP_SECONDS = float(os.getenv("WORKER_REF_SKIP_SECONDS", "30.0"))  # skip intros/titles

MAX_ATEMPO = float(os.getenv("WORKER_MAX_ATEMPO", "2.0"))
DISPATCH_POLL_INTERVAL = float(os.getenv("WORKER_DISPATCH_POLL_INTERVAL", "1.0"))

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("WORKER_ALLOWED_ORIGINS", "*").split(",")]
