"""SQLite-backed job store, keyed by (mkv_sha256, language).

Same pattern as gateway/app/db.py: a job id is "<hash>:<language>", resubmitting
the same mkv content + language returns the existing job instead of
reprocessing it. States:

    queued      -> accepted, waiting for the single-slot worker loop
    processing  -> worker loop picked it up, TTS/assembly running
    done        -> output_path has the finished audio file
    failed      -> error has the reason

Crash recovery mirrors the gateway: 'processing' only exists while this
process is alive; a restart demotes it back to 'queued'.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    mkv_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error TEXT,
    output_path TEXT,
    UNIQUE(mkv_hash, language)
);
"""


@dataclass
class Job:
    id: str
    mkv_hash: str
    language: str
    status: str
    created_at: float
    updated_at: float
    error: Optional[str]
    output_path: Optional[str]


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(**{k: row[k] for k in row.keys()})


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(SCHEMA)


def reconcile_stale_jobs() -> list[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT id FROM jobs WHERE status='processing'").fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            conn.execute(
                f"UPDATE jobs SET status='queued', updated_at=? WHERE id IN ({','.join('?' * len(ids))})",
                [time.time(), *ids],
            )
        return ids


def create_or_get_job(mkv_hash: str, language: str) -> tuple[Job, bool]:
    now = time.time()
    job_id = f"{mkv_hash}:{language}"
    with _connect() as conn:
        existing = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if existing:
            return _row_to_job(existing), False
        conn.execute(
            "INSERT INTO jobs (id, mkv_hash, language, status, created_at, updated_at, error, output_path) "
            "VALUES (?, ?, ?, 'queued', ?, ?, NULL, NULL)",
            (job_id, mkv_hash, language, now, now),
        )
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row), True


def get_job(job_id: str) -> Optional[Job]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None


def claim_next_queued_job() -> Optional[Job]:
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE jobs SET status='processing', updated_at=? WHERE id=? AND status='queued'",
            (now, row["id"]),
        )
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        return _row_to_job(row)


def set_status(job_id: str, status: str, error: Optional[str] = None, output_path: Optional[str] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, error=?, output_path=COALESCE(?, output_path), updated_at=? WHERE id=?",
            (status, error, output_path, time.time(), job_id),
        )
