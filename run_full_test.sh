#!/usr/bin/env bash
# Run the jelly-dub worker end-to-end against a real mkv+srt pair and save
# the resulting dubbed audio track next to the input.
#
# Usage:
#   ./run_full_test.sh <file.mkv> <subtitles.srt> [language] [output.wav]
#
# Must run on a box with the Arc GPU passthrough available (bazzite), from
# inside the jelly-dub repo checkout (expects ./worker/Dockerfile to exist).
# Starts (or reuses) a long-lived worker container bound to 127.0.0.1 only -
# this is LAN/local-only tooling, not meant to be exposed publicly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$SCRIPT_DIR/worker"
IMAGE="localhost/jelly-dub-worker"
CONTAINER="jelly-dub-worker-run"
PORT=8420
DATA_DIR="$SCRIPT_DIR/.run_full_test_data"

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
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  status: $STATUS"
  case "$STATUS" in
    done) break ;;
    failed)
      ERROR=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error'))")
      echo "job failed: $ERROR" >&2
      exit 1
      ;;
  esac
  sleep 5
done

curl -s "http://127.0.0.1:${PORT}/dub/${JOB_ID}/audio" -o "$OUT"
echo "wrote $OUT"
