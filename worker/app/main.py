"""Dub worker: FastAPI + SQLite, single GPU-bound dispatch slot.

  POST /dub            multipart: mkv=<file>, srt=<file>, language=<str>
                        -> 202 {id, mkv_hash, language, status: "queued"}
                        id is "<sha256 of mkv content>:<language>" so the
                        same video re-submitted (even under a different
                        filename) in the same language is deduped instead of
                        reprocessed. A different language for an already-
                        uploaded video reuses the stored mkv.

  GET  /dub/{job_id}   -> status JSON (queued/processing/done/failed)

  GET  /dub/{job_id}/audio
                        -> the generated audio file once status is "done"
                        (404 if unknown, 409 if not finished yet)

Unauthenticated by the same design decision as the gateway: this is meant to
be called by the gateway (or directly, for testing) on the trusted LAN, not
exposed publicly. See gateway/app/main.py's docstring for the full reasoning.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config, db, dispatcher


class JobResponse(BaseModel):
    id: str
    mkv_hash: str
    language: str
    status: str
    created_at: float
    updated_at: float
    error: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: int | None = None  # None until progress_total is known

    @classmethod
    def from_job(cls, job: db.Job) -> "JobResponse":
        return cls(
            id=job.id, mkv_hash=job.mkv_hash, language=job.language,
            status=job.status, created_at=job.created_at, updated_at=job.updated_at,
            error=job.error, progress_current=job.progress_current,
            progress_total=job.progress_total, progress_percent=job.progress_percent,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    reconciled = db.reconcile_stale_jobs()
    print(f"[startup] reconciled {len(reconciled)} stale job(s): {reconciled}", flush=True)
    task = asyncio.create_task(dispatcher.run_forever())
    try:
        yield
    finally:
        dispatcher.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="jelly-dub worker", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _hash_and_save(upload: UploadFile, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with dest.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            hasher.update(chunk)
            f.write(chunk)
    return hasher.hexdigest()


@app.post("/dub", response_model=JobResponse, status_code=202)
def submit_dub(
    mkv: UploadFile = File(...),
    srt: UploadFile = File(...),
    language: str = Form(...),
) -> JobResponse:
    work_root = Path(config.WORK_DIR)

    # Hash first into a temp spot so we don't claim a hash-named dir for
    # content we haven't verified yet.
    tmp_path = work_root / "_tmp_upload.mkv"
    mkv_hash = _hash_and_save(mkv, tmp_path)

    job, created = db.create_or_get_job(mkv_hash, language)
    job_dir = work_root / job.id.replace(":", "_")
    job_dir.mkdir(parents=True, exist_ok=True)

    mkv_store_dir = work_root / mkv_hash
    mkv_store_path = mkv_store_dir / "input.mkv"
    if not mkv_store_path.exists():
        mkv_store_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), str(mkv_store_path))
    else:
        tmp_path.unlink(missing_ok=True)

    if created:
        srt_dest = job_dir / "input.srt"
        with srt_dest.open("wb") as f:
            shutil.copyfileobj(srt.file, f)

    return JobResponse.from_job(job)


@app.get("/dub/{job_id}", response_model=JobResponse)
def get_dub(job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse.from_job(job)


@app.get("/dub/{job_id}/audio")
def get_dub_audio(job_id: str) -> FileResponse:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "done" or not job.output_path:
        raise HTTPException(status_code=409, detail=f"job status is '{job.status}', not ready yet")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=410, detail="output file missing")
    return FileResponse(path, media_type="audio/wav", filename=f"{job.language}_dub.wav")
