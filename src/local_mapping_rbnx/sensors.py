# SPDX-License-Identifier: MulanPSL-2.0
"""Pure sensor decoding helpers.

ROS messages are converted to plain numpy arrays by the node before calling
these functions, keeping geometry testable without rclpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_POINT_FIELD_FLOAT32 = 7
_POINT_FIELD_FLOAT64 = 8


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera_info(cls, msg) -> "Intrinsics":
        width = int(msg.width)
        height = int(msg.height)
        k = list(msg.k)
        if len(k) < 9 or width <= 0 or height <= 0:
            raise ValueError("CameraInfo is missing a usable K matrix")
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            raise ValueError("CameraInfo intrinsics are invalid")
        return cls(width, height, fx, fy, cx, cy)


def depth_to_metres(
    data: bytes | bytearray | memoryview,
    encoding: str,
    height: int,
    width: int,
    step: int,
) -> np.ndarray:
    if encoding in ("32FC1", "32FC1_LE"):
        dtype = np.dtype("<f4")
    elif encoding in ("16UC1", "mono16"):
        dtype = np.dtype("<u2")
    else:
        raise ValueError(f"unsupported depth encoding: {encoding}")
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError("depth image payload is shorter than height*step")
    rows = raw[:expected].reshape(height, step)
    values = rows[:, : width * dtype.itemsize].copy().view(dtype).reshape(height, width)
    depth = values.astype(np.float32)
    if dtype == np.dtype("<u2"):
        depth /= 1000.0
    return depth


def backproject_depth(
    depth: np.ndarray,
    intr: Intrinsics,
    stride: int,
    min_range_m: float,
    max_range_m: float,
) -> np.ndarray:
    """Return Nx3 points in the camera optical frame (right/down/forward)."""
    if depth.ndim != 2:
        raise ValueError("depth image must be 2-D")
    step = max(1, int(stride))
    z = depth[::step, ::step]
    u = np.arange(0, depth.shape[1], step, dtype=np.float32)
    v = np.arange(0, depth.shape[0], step, dtype=np.float32)
    if u.size != z.shape[1] or v.size != z.shape[0]:
        # Non-divisible dimensions still follow numpy slice semantics.
        u = u[: z.shape[1]]
        v = v[: z.shape[0]]
    uu, vv = np.meshgrid(u, v)
    valid = np.isfinite(z) & (z >= min_range_m) & (z <= max_range_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    zz = z[valid].astype(np.float64)
    x = (uu[valid].astype(np.float64) - intr.cx) * zz / intr.fx
    y = (vv[valid].astype(np.float64) - intr.cy) * zz / intr.fy
    return np.stack((x, y, zz), axis=1)


def scan_endpoints(
    ranges: list[float] | tuple[float, ...],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    finite = np.isfinite(values) & (values >= range_min) & (values <= range_max)
    if not np.any(finite):
        return np.zeros((0, 3), dtype=np.float64)
    angles = angle_min + np.arange(values.size, dtype=np.float64) * angle_increment
    r = values[finite]
    a = angles[finite]
    return np.stack((r * np.cos(a), r * np.sin(a), np.zeros_like(r)), axis=1)


def pointcloud_xyz(
    data: bytes | bytearray | memoryview,
    point_step: int,
    width: int,
    height: int,
    row_step: int,
    fields,
    is_bigendian: bool,
) -> np.ndarray:
    """Decode x/y/z from a PointCloud2 payload without sensor_msgs_py."""
    by_name = {str(field.name): field for field in fields}
    for name in ("x", "y", "z"):
        if name not in by_name:
            raise ValueError(f"PointCloud2 is missing the {name} field")
        if int(by_name[name].datatype) not in (_POINT_FIELD_FLOAT32, _POINT_FIELD_FLOAT64):
            raise ValueError(f"PointCloud2 {name} field must be float32 or float64")
    endian = ">" if is_bigendian else "<"
    names = ["x", "y", "z"]
    formats = [
        endian + ("f4" if int(by_name[n].datatype) == _POINT_FIELD_FLOAT32 else "f8")
        for n in names
    ]
    offsets = [int(by_name[n].offset) for n in names]
    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    raw = np.frombuffer(data, dtype=np.uint8)
    if point_step <= 0 or row_step < width * point_step:
        raise ValueError("PointCloud2 row layout is invalid")
    expected = row_step * height
    if raw.size < expected:
        raise ValueError("PointCloud2 payload is shorter than row_step*height")
    if row_step == width * point_step:
        points = np.frombuffer(
            raw[: width * height * point_step].tobytes(), dtype=dtype, count=width * height
        )
    else:
        rows = raw[:expected].reshape(height, row_step)
        points = np.concatenate(
            [
                np.frombuffer(rows[r, : width * point_step].tobytes(), dtype=dtype)
                for r in range(height)
            ]
        )
    out = np.stack([points[n].astype(np.float64) for n in names], axis=1)
    return out[np.isfinite(out).all(axis=1)]


def nearest_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB image must be HxWx3+")
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("RGB image is empty")
    ys = (np.arange(height, dtype=np.float64) * (src_h / height)).astype(np.int64)
    xs = (np.arange(width, dtype=np.float64) * (src_w / width)).astype(np.int64)
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)
    return image[ys[:, None], xs[None, :], :3]


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)[None, ...]


def decode_rgb(
    data: bytes | bytearray | memoryview,
    encoding: str,
    height: int,
    width: int,
    step: int,
) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError("RGB image payload is shorter than height*step")
    rows = raw[:expected].reshape(height, step)
    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    else:
        raise ValueError(f"unsupported RGB encoding: {encoding}")
    image = rows[:, : width * channels].reshape(height, width, channels)
    if encoding.startswith("bgr"):
        image = image[:, :, [2, 1, 0, 3]] if channels == 4 else image[:, :, ::-1]
    return image[:, :, :3].copy()