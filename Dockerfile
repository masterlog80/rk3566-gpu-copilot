# ── RK3566 NPU Stress Test – Docker Image ───────────────────────────────────
#
# Target: arm64v8 (Rockchip RK3566 / Orange Pi CM4)
#
# Build:
#   docker buildx build --platform linux/arm64 -t rk3566-npu-stress:latest .
#
# Run (on target hardware):
#   docker run -d --privileged --device /dev/dri -p 8080:8080 rk3566-npu-stress:latest
#
# Run (dev/simulation on any host):
#   docker run -d -p 8080:8080 rk3566-npu-stress:latest
# ─────────────────────────────────────────────────────────────────────────────

FROM arm64v8/ubuntu:22.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        gcc \
        libgomp1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app
COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Try to install rknn-toolkit-lite2 (optional, for actual hardware) ────────
# The package is architecture-specific and may not be on PyPI for all versions.
# A failure here is non-fatal – the app will run in simulation mode instead.
RUN pip3 install --no-cache-dir rknn-toolkit-lite2 || \
    echo "[INFO] rknn-toolkit-lite2 not available – simulation mode will be used"

# ── RKNN native runtime library (librknnrt.so) ───────────────────────────────
# Required for real NPU inference on RK3566/RK3568 hardware.
# Downloaded from the official Rockchip RKNN-Toolkit2 repository.
# Pin to v2.3.2 – update RKNN_TOOLKIT2_TAG to match the rknn-toolkit-lite2
# version installed above when upgrading.
#
# Alternative: if the host already has the library you can skip the download
# and map it into the container at runtime instead:
#   -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so
ARG RKNN_TOOLKIT2_TAG=v2.3.2
RUN wget -q \
      "https://github.com/airockchip/rknn-toolkit2/raw/${RKNN_TOOLKIT2_TAG}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" \
      -O /usr/lib/librknnrt.so && \
    ldconfig || \
    echo "[WARN] librknnrt.so download failed – container will fall back to simulation mode on hardware"

# ── RKNN model (MobileNetV1, ~4 MB) ─────────────────────────────────────────
# Downloaded from the official Rockchip RKNN-Toolkit2 examples repository.
# Used for NPU inference stress-testing.  Not required in simulation mode.
RUN mkdir -p /app/models && \
    wget -q \
      "https://github.com/airockchip/rknn-toolkit2/raw/${RKNN_TOOLKIT2_TAG}/rknn-toolkit-lite2/examples/rknn_mobilenet_demo/model/RK356X/mobilenet_v1.rknn" \
      -O /app/models/mobilenet_v1.rknn || \
    echo "[INFO] Model download failed – simulation mode will be used"

# ── Application code ─────────────────────────────────────────────────────────
COPY app/ .

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8080

ENV MODEL_PATH=/app/models/mobilenet_v1.rknn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
