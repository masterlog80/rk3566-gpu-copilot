#!/usr/bin/env bash
# Entrypoint for the RK3566 NPU Stress Test container.
#
# Mounts debugfs if it is not already mounted so that the NPU utilisation
# counter at /sys/kernel/debug/rknpu/load is accessible.
# This requires the container to be started with --privileged (or at minimum
# CAP_SYS_ADMIN).  When running in simulation / dev mode without privileges
# the mount will silently fail and the app falls back to reporting 0 % util.

set -e

if ! mountpoint -q /sys/kernel/debug 2>/dev/null; then
    mount -t debugfs debugfs /sys/kernel/debug 2>/dev/null \
        && echo "[INFO] entrypoint: debugfs mounted at /sys/kernel/debug" \
        || echo "[WARN] entrypoint: could not mount debugfs – NPU utilisation will be unavailable (is the container running with --privileged?)"
else
    echo "[INFO] entrypoint: debugfs already mounted"
fi

exec uvicorn main:app --host 0.0.0.0 --port 8080
