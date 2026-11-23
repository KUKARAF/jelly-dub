# Intel Arc (Battlemage/B70) GPU runtime: the stock PyTorch XPU wheels do NOT
# bundle the Level-Zero loader/ICD the way earlier docs implied - without it
# torch.xpu.is_available() silently returns False (device count 0) and falls
# back to CPU. ghcr.io/ggml-org/llama.cpp:server-intel is a verified-working
# SYCL/Level-Zero image on this exact GPU; borrow its runtime libs instead of
# re-deriving the correct Intel apt repo/package set from scratch.
FROM ghcr.io/ggml-org/llama.cpp:server-intel AS intel-gpu-runtime

FROM docker.io/library/python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=intel-gpu-runtime /usr/lib/x86_64-linux-gnu/libze_loader.so* /usr/lib/x86_64-linux-gnu/
COPY --from=intel-gpu-runtime /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so* /usr/lib/x86_64-linux-gnu/
COPY --from=intel-gpu-runtime /etc/OpenCL/vendors/intel.icd /etc/OpenCL/vendors/intel.icd
COPY --from=intel-gpu-runtime /etc/OpenCL/vendors/intel64.icd /etc/OpenCL/vendors/intel64.icd
RUN ldconfig

# chatterbox-tts pins torch==2.6.0 for python<3.14, but Intel's XPU wheel index
# only ships torch>=2.9.0+xpu (older ones were pulled). Python 3.14 takes
# chatterbox-tts's torch>=2.9.0 branch instead, which the XPU index satisfies.
# Pin the matched XPU triple exactly and constrain chatterbox-tts's own
# resolve so it can't swap in a mismatched non-xpu build (torch/torchvision
# ABI mismatch -> "operator torchvision::nms does not exist").
RUN pip install --no-cache-dir \
    torch==2.9.0+xpu torchvision==0.24.0+xpu torchaudio==2.9.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu

RUN printf "torch==2.9.0+xpu\ntorchvision==0.24.0+xpu\ntorchaudio==2.9.0+xpu\n" > /tmp/constraints.txt && \
    pip install --no-cache-dir chatterbox-tts \
    --constraint /tmp/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/xpu

WORKDIR /work
