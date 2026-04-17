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

# ── RKNN model (ResNet18 for RK3566/RK3568, ~11 MB) ─────────────────────────
# Downloaded from the official Rockchip RKNN-Toolkit2 examples repository.
# Used for NPU inference stress-testing.  Not required in simulation mode.
#
# Note: use raw.githubusercontent.com (not github.com/raw) so that files
# stored with Git LFS are served as the real binary, not an LFS pointer.
# A minimum-size check ensures a failed/partial download is caught early
# rather than producing a misleading RKNN_ERR_MODEL_INVALID at runtime.
RUN mkdir -p /app/models && \
    wget -q \
      "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/${RKNN_TOOLKIT2_TAG}/rknn-toolkit-lite2/examples/resnet18/resnet18_for_rk3566_rk3568.rknn" \
      -O /app/models/resnet18_for_rk3566_rk3568.rknn && \
    [ "$(stat -c%s /app/models/resnet18_for_rk3566_rk3568.rknn)" -gt 10000000 ] || \
    { rm -f /app/models/resnet18_for_rk3566_rk3568.rknn; \
      echo "[INFO] ResNet18 model download failed – simulation mode will be used"; }

# ── RKNN model (MobileNetV1 for RK3566/RK3568, ~4 MB) ───────────────────────
RUN wget -q \
      "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/${RKNN_TOOLKIT2_TAG}/rknn-toolkit-lite2/examples/mobilenet_v1/mobilenet_v1_for_rk3566_rk3568.rknn" \
      -O /app/models/mobilenet_v1_for_rk3566_rk3568.rknn && \
    [ "$(stat -c%s /app/models/mobilenet_v1_for_rk3566_rk3568.rknn)" -gt 3000000 ] || \
    { rm -f /app/models/mobilenet_v1_for_rk3566_rk3568.rknn; \
      echo "[INFO] MobileNetV1 model download failed – this test type will fall back to simulation"; }

# ── RKNN model (MobileNetV2 for RK3566/RK3568, ~4 MB) ───────────────────────
RUN wget -q \
      "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/${RKNN_TOOLKIT2_TAG}/rknn-toolkit-lite2/examples/mobilenet_v2/mobilenet_v2_for_rk3566_rk3568.rknn" \
      -O /app/models/mobilenet_v2_for_rk3566_rk3568.rknn && \
    [ "$(stat -c%s /app/models/mobilenet_v2_for_rk3566_rk3568.rknn)" -gt 3000000 ] || \
    { rm -f /app/models/mobilenet_v2_for_rk3566_rk3568.rknn; \
      echo "[INFO] MobileNetV2 model download failed – this test type will fall back to simulation"; }

# ── Application code ─────────────────────────────────────────────────────────
COPY app/ .

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8080

ENV MODEL_PATH=/app/models/resnet18_for_rk3566_rk3568.rknn

# The entrypoint mounts debugfs (needed for /sys/kernel/debug/rknpu/load) when
# the container has CAP_SYS_ADMIN / --privileged, then launches uvicorn.
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
