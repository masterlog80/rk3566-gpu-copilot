# RK3566 NPU Stress Test Dashboard

A self-contained Docker image that provides a **modern web UI** to start, stop, and monitor a stress test on the NPU of a Rockchip **RK3566** (Orange Pi CM4).

---

## Features

| Feature | Details |
|---|---|
| **Modern Web UI** | Responsive dark-theme dashboard with real-time charts |
| **Configurable duration** | Slider + preset buttons (30 s → 5 min), 5–3600 s range |
| **Start / Stop control** | One-click start and emergency stop |
| **Live metrics** | FPS, inference latency, NPU utilisation, SoC temperature |
| **Real-time chart** | FPS, temperature, and NPU % plotted over time |
| **Results summary** | Avg FPS, peak temp, avg latency, total inferences |
| **Hardware + simulation** | Runs real NPU inference via `rknnlite2`; falls back to a realistic simulation if the RKNN runtime is unavailable |

---

## Screenshots

![RK3566 NPU Stress Test Dashboard](https://github.com/user-attachments/assets/90c05f5a-2853-48e8-a878-2bccfbb5d262)

---

## Quick Start

### Prerequisites

- Docker ≥ 24 with `buildx` (for cross-compilation on x86)
- An Orange Pi CM4 (or any RK3566 board) running a Linux image with Rockchip BSP kernel for hardware NPU access

### Build the image

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/masterlog80/rk3566-gpu-copilot.git
cd rk3566-gpu-copilot

# 2. Build and start the container (first run builds the image)
yes | docker image prune --all
docker build -t rk3566-npu-stress .
#docker compose up -d --build

# 3. Deploy the composer file:
docker compose -f docker-compose.yml up -d --remove-orphans

# 4. Or cross-compile from an x86 host:
docker buildx build --platform linux/arm64 \
  -t rk3566-npu-stress:latest \
  --load .

# 5. Open the dashboard
```

Or manually:

```bash
docker run -d \
  --privileged \
  --device /dev/dri \
  -p 8080:8080 \
  rk3566-npu-stress:latest
```

Open **http://\<board-ip\>:8080** in your browser.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Docker Container (arm64v8/ubuntu:22.04)            │
│                                                     │
│  ┌──────────────┐     ┌───────────────────────────┐ │
│  │  FastAPI     │────▶│  NPUStressTest worker     │ │
│  │  (uvicorn)   │     │  - rknnlite2 primary mode │ │
│  │  :8080       │     │  - simulation fallback    │ │
│  └──────┬───────┘     └───────────────────────────┘ │
│         │ SSE / REST                                 │
│  ┌──────▼───────────────────────────────────────┐   │
│  │  Web UI (Tailwind + Alpine.js + Chart.js)    │   │
│  └─────────────────────────────────────────────-┘   │
└─────────────────────────────────────────────────────┘
```

**Endpoints**

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Web dashboard |
| `POST` | `/api/start?duration=60` | Start stress test |
| `POST` | `/api/stop` | Stop running test |
| `GET`  | `/api/status` | JSON status snapshot |
| `GET`  | `/api/events` | SSE stream (500 ms interval) |

---


### Run (simulation / development mode – any host)

```bash
docker run -d -p 8080:8080 rk3566-npu-stress:latest
```

The app automatically detects that `rknnlite2` is unavailable and falls back to a realistic simulated workload.  The UI and all metrics work identically.

---

## File Structure

```
.
├── Dockerfile            # arm64v8/ubuntu:22.04 based image
├── docker-compose.yml    # Easy deployment with device mapping
├── app/
│   ├── main.py           # FastAPI application & API endpoints
│   ├── npu_stress.py     # NPU stress-test worker (rknnlite2 + simulation)
│   ├── requirements.txt  # Python dependencies
│   └── static/
│       └── index.html    # Single-page dashboard
└── README.md
```

---

## NPU Hardware Notes

### Device access
The container needs privileged access or specific device mappings to reach the Rockchip RKNPU2 driver:

```bash
--privileged          # full access, simplest option
--device /dev/dri     # Mali/NPU DRM device node
```

`/sys/kernel/debug/rknpu/load` (NPU utilisation) requires `debugfs` to be mounted and `CAP_SYS_ADMIN`.

### librknnrt.so – Rockchip NPU runtime library

The image downloads `librknnrt.so` from the official
[airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64)
repository into `/usr/lib/` at **build time**.

If the build-time download failed (e.g. no internet access on the build host), or if you
see the following error in the container logs, the library is missing:

```
!!! It is detected that /usr/lib/librknnrt.so is missing in the container.
```

**Option A – rebuild the image** (recommended, requires internet on build host):

```bash
docker buildx build --platform linux/arm64 -t rk3566-npu-stress:latest .
```

**Option B – copy the library manually** onto the board and mount it into the container:

```bash
# On the board, download once:
wget -O /usr/lib/librknnrt.so \
  "https://github.com/airockchip/rknn-toolkit2/raw/v2.3.2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

# Then run with a volume mount:
docker run -d \
  --privileged \
  --device /dev/dri \
  -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so \
  -p 8080:8080 \
  rk3566-npu-stress:latest
```

When using `docker compose`, uncomment the matching volume line in `docker-compose.yml`.

Without `librknnrt.so` the application automatically falls back to **simulation mode**
(the UI is fully functional, but no real NPU inference is performed).

### Supported chips
The `resnet18_for_rk3566_rk3568.rknn` model bundled in the image is compiled for the **RK356X** (RK3566 / RK3568) NPU target.  It will **not** run on RK3588 or RK3399 without recompilation.

### Custom model
Mount a different `.rknn` model and point to it via the `MODEL_PATH` environment variable:

```bash
docker run -d \
  --privileged \
  --device /dev/dri \
  -e MODEL_PATH=/models/my_model.rknn \
  -v /path/to/models:/models \
  -p 8080:8080 \
  rk3566-npu-stress:latest
```
