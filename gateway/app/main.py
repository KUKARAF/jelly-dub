"""Dub-job gateway: FastAPI + SQLite.

Four endpoints:
  GET  /health       liveness check
  POST /jobs        submit a dub job for (item_id, language), deduped
  GET  /jobs/{id}    status of one job
  GET  /jobs         list all jobs

INTENTIONALLY UNAUTHENTICATED. This exists so a button injected into the
Jellyfin web client (a browser page, not a trusted backend) can call it
directly with no token to manage - the whole point is to make baking this
into Jellyfin's UI simple. jellyfin_client.check_user_auth() still exists
and is NOT called from any endpoint here; wiring it back in is a one-line
addition to create_job() if real per-user auth is ever needed.

The real safety boundary is network placement, not this code: this service
must only be reachable from the trusted household LAN (no public port, no
reverse-proxy exposure), the same way qwen-tts/litellm/etc. are not
published - see osmosis compose conventions. Anyone who can reach this port
can submit jobs and occupy the single GPU worker slot. Do not deploy this
behind a public-facing proxy without adding real auth first.

On startup: init the DB, reconcile any job left in 'waking'/'running' from a
previous crash back to 'queued', then start the single-slot dispatcher loop.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config, db, dispatcher


class JobRequest(BaseModel):
    item_id: str
    language: str


class JobResponse(BaseModel):
    id: str
    item_id: str
    language: str
    status: str
    created_at: float
    updated_at: float
    attempts: int
    error: str | None = None

    @classmethod
    def from_job(cls, job: db.Job) -> "JobResponse":
        return cls(**job.__dict__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    reconciled = db.reconcile_stale_jobs()
    if reconciled:
        print(f"[startup] reconciled {len(reconciled)} stale job(s) back to queued: {reconciled}", flush=True)
    else:
        print("[startup] no stale jobs to reconcile", flush=True)

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


app = FastAPI(title="jelly-dub gateway", lifespan=lifespan)

# Allow the browser-injected button to call this cross-origin from the
# Jellyfin web client's own domain. Wide open by default to match the "no
# auth, LAN-only" design above; narrow to config.ALLOWED_ORIGINS once a real
# Jellyfin domain is pinned, instead of relying on this being permissive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobResponse)
def create_job(req: JobRequest) -> JobResponse:
    job, created = db.create_or_get_job(req.item_id, req.language)
    return JobResponse.from_job(job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResponse.from_job(job)


@app.get("/jobs", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    return [JobResponse.from_job(j) for j in db.list_jobs()]
