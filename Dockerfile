FROM docker.io/library/python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Stock PyTorch XPU wheels bundle the Level-Zero runtime needed for Intel Arc
# (Battlemage/B70). Host only needs the xe kernel driver + GuC/HuC firmware
# and /dev/dri passed through - no oneAPI basekit install required.
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/xpu

RUN pip install --no-cache-dir chatterbox-tts

WORKDIR /work
