# Builds on the already-validated GPU image (Level-Zero + IGC runtime libs,
# torch 2.9.0+xpu, chatterbox-tts) rather than re-deriving the GPU setup.
FROM ghcr.io/kukaraf/jelly-dub:latest

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app /app/app
WORKDIR /app

ENV WORKER_WORK_DIR=/work/jobs
ENV WORKER_DB_PATH=/work/data/worker.sqlite3

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8420"]
