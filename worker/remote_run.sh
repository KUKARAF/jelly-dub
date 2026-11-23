#!/usr/bin/env bash
# Runs ON the GPU box (bazzite): build/start the worker, submit one job,
# poll it, and leave the output audio at a known path. This is the part
# run_full_test.sh (the local-facing entrypoint) copies over and invokes via
# ssh - it is not meant to be run manually except for debugging on the box.
#
# Usage: remote_run.sh <mkv> <srt> <language> <output-wav>
# Paths are relative to wherever this script lives (expects ./Dockerfile and
# ./app next to it, i.e. this file must stay inside worker/).

set -euo pipefail

WORKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="localhost/jelly-dub-worker"
CONTAINER="jelly-dub-worker-run"
PORT=8420
DATA_DIR="$WORKER_DIR/.run_full_test_data"

MKV="$1"
SRT="$2"
LANG="$3"
OUT="$4"

echo "[1/5] ensuring worker image is built..."
if ! podman image exists "$IMAGE"; then
  podman build -t "$IMAGE" -f "$WORKER_DIR/Dockerfile" "$WORKER_DIR"
fi

echo "[2/5] ensuring worker container is running..."
mkdir -p "$DATA_DIR"
if ! podman container exists "$CONTAINER" || [ "$(podman inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != "true" ]; then
  podman rm -f "$CONTAINER" >/dev/null 2>&1 || true
  podman run -d --name "$CONTAINER" \
    --device /dev/dri --group-add keep-groups --security-opt label=disable --ipc host \
    -v "$DATA_DIR:/work" \
    -p "127.0.0.1:${PORT}:${PORT}" \
    "$IMAGE" >/dev/null
fi

echo "[3/5] waiting for worker health..."
for i in $(seq 1 60); do
  if curl -s -f "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  if [ "$i" -eq 60 ]; then
    echo "error: worker did not become healthy in time; check: podman logs $CONTAINER" >&2
    exit 1
  fi
done

echo "[4/5] submitting job (mkv=$MKV srt=$SRT language=$LANG)..."
RESPONSE=$(curl -s -X POST "http://127.0.0.1:${PORT}/dub" \
  -F "mkv=@${MKV}" \
  -F "srt=@${SRT}" \
  -F "language=${LANG}")
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "job id: $JOB_ID"

echo "[5/5] waiting for the job to finish (this can take a while for a full episode)..."
while true; do
  STATUS_JSON=$(curl -s "http://127.0.0.1:${PORT}/dub/${JOB_ID}")
  IFS=$'\t' read -r STATUS DISPLAY ERROR < <(echo "$STATUS_JSON" | python3 -c "
import sys, json
j = json.load(sys.stdin)
status, pct, cur, total = j['status'], j.get('progress_percent'), j['progress_current'], j['progress_total']
display = f'processing {pct}% done ({cur}/{total} cues)' if status == 'processing' and pct is not None else status
print(f\"{status}\t{display}\t{j.get('error') or ''}\")
")
  echo "  $DISPLAY"
  case "$STATUS" in
    done) break ;;
    failed)
      echo "job failed: $ERROR" >&2
      exit 1
      ;;
  esac
  sleep 5
done

curl -s "http://127.0.0.1:${PORT}/dub/${JOB_ID}/audio" -o "$OUT"
echo "wrote $OUT"
