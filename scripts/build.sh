#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# local_mapping_rbnx build phase: generate Robonix gRPC stubs and verify the
# host runtime. The service is native-only; it has no container image.
set -euo pipefail

PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PKG"
mkdir -p rbnx-build/data

# The runtime imports rclpy/tf2_ros, which only exist after sourcing ROS.
ROS_SETUP="${RBNX_MINECRAFT_ROS_SETUP:-}"
if [[ -z "$ROS_SETUP" ]]; then
    for candidate in "/opt/ros/${ROS_DISTRO:-lyrical}/setup.bash" /opt/ros/lyrical/setup.bash /opt/ros/jazzy/setup.bash /opt/ros/humble/setup.bash; do
        if [[ -f "$candidate" ]]; then ROS_SETUP="$candidate"; break; fi
    done
fi
if [[ -z "$ROS_SETUP" || ! -f "$ROS_SETUP" ]]; then
    echo "[local-mapping/build] ROS 2 setup not found; set RBNX_MINECRAFT_ROS_SETUP" >&2
    exit 2
fi
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

if command -v rbnx >/dev/null 2>&1; then
    FLAGS=()
    [[ "${RBNX_BUILD_CLEAN:-}" == "1" ]] && FLAGS+=(--clean)
    echo "[local-mapping/build] rbnx codegen ${FLAGS[*]:-}"
    rbnx codegen -p "$PKG" "${FLAGS[@]}"
else
    echo "[local-mapping/build] WARNING: rbnx not in PATH; skipping codegen" >&2
fi

VENV="$PKG/rbnx-build/codegen-venv"
PYBIN="${LOCAL_MAPPING_PYTHON:-$VENV/bin/python3}"
if [[ -z "${LOCAL_MAPPING_PYTHON:-}" ]]; then
    if [[ ! -x "$PYBIN" ]]; then
        echo "[local-mapping/build] create runtime venv"
        python3 -m venv --system-site-packages "$VENV"
    fi
    if ! "$PYBIN" -c 'import grpc, google.protobuf, mcp, fastmcp' >/dev/null 2>&1; then
        echo "[local-mapping/build] install runtime dependencies"
        "$PYBIN" -m pip install --no-cache-dir --disable-pip-version-check \
            grpcio==1.80.0 protobuf==6.33.6 'mcp>=1.28.1,<2' 'fastmcp>=3,<4'
    fi
fi
"$PYBIN" - <<'PY'
import importlib
for name in ("grpc", "numpy", "yaml", "rclpy", "tf2_ros", "sensor_msgs", "nav_msgs"):
    importlib.import_module(name)
print("[local-mapping/build] runtime imports OK")
PY

if "$PYBIN" -c 'import onnxruntime' >/dev/null 2>&1; then
    echo "[local-mapping/build] onnxruntime present (semantics available)"
else
    echo "[local-mapping/build] onnxruntime absent; semantics stay disabled" >&2
fi

touch rbnx-build/.rbnx-built
echo "[local-mapping/build] done"