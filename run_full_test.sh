#!/usr/bin/env bash
# Run the jelly-dub worker against a real mkv+srt pair, executing remotely on
# the GPU box (bazzite) over ssh - this is meant to be run from a dev
# machine, not on bazzite itself.
#
# Usage:
#   ./run_full_test.sh <file.mkv> <subtitles.srt> [language] [output.wav]
#
# What it does: scp's worker/ (the Dockerfile + app + remote_run.sh) and the
# input files to the remote box, ssh's in to build/start the worker and run
# one job there (all GPU/podman work happens on bazzite, nothing locally),
# then scp's the resulting dubbed audio back to the local output path.
#
# Override the remote target with env vars:
#   JELLY_DUB_HOST (ssh destination, default: local_bazzite)
#   JELLY_DUB_REMOTE_DIR (where to sync worker/ on the remote box,
#                         default: ~/jelly-dub-worker)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$SCRIPT_DIR/worker"
HOST="${JELLY_DUB_HOST:-local_bazzite}"

# Resolve to an absolute path up front: paths get passed through ssh/curl's
# @filename args later, where a literal unexpanded "~" (e.g. from
# single-quoting in a remote ssh command) makes curl fail with "couldn't
# read file" (exit 26) instead of expanding to the home directory.
REMOTE_HOME="$(ssh "$HOST" 'echo $HOME')"
REMOTE_DIR="${JELLY_DUB_REMOTE_DIR:-${REMOTE_HOME}/jelly-dub-worker}"

if [ $# -lt 2 ]; then
  echo "usage: $0 <file.mkv> <subtitles.srt> [language] [output.wav]" >&2
  exit 1
fi

MKV="$1"
SRT="$2"
LANG="${3:-pl}"
OUT="${4:-${MKV%.*}.${LANG}.dub.wav}"

for f in "$MKV" "$SRT"; do
  if [ ! -f "$f" ]; then
    echo "error: file not found: $f" >&2
    exit 1
  fi
done

if ! [ -d "$WORKER_DIR" ]; then
  echo "error: expected worker/ next to this script ($WORKER_DIR not found)" >&2
  exit 1
fi

echo "[local 1/4] syncing worker/ to ${HOST}:${REMOTE_DIR}..."
ssh "$HOST" "mkdir -p ${REMOTE_DIR}/input"
scp -rq "$WORKER_DIR/Dockerfile" "$WORKER_DIR/requirements.txt" "$WORKER_DIR/app" "$WORKER_DIR/remote_run.sh" \
  "$HOST:${REMOTE_DIR}/"
ssh "$HOST" "chmod +x ${REMOTE_DIR}/remote_run.sh"

echo "[local 2/4] copying input files to ${HOST}..."
REMOTE_MKV="${REMOTE_DIR}/input/$(basename "$MKV")"
REMOTE_SRT="${REMOTE_DIR}/input/$(basename "$SRT")"
REMOTE_OUT="${REMOTE_DIR}/input/output_${LANG}.wav"
scp -q "$MKV" "$HOST:${REMOTE_MKV}"
scp -q "$SRT" "$HOST:${REMOTE_SRT}"

echo "[local 3/4] running the pipeline remotely on ${HOST} (this can take a while)..."
ssh "$HOST" "${REMOTE_DIR}/remote_run.sh '${REMOTE_MKV}' '${REMOTE_SRT}' '${LANG}' '${REMOTE_OUT}'"

echo "[local 4/4] fetching result..."
scp -q "$HOST:${REMOTE_OUT}" "$OUT"
echo "wrote $OUT"
