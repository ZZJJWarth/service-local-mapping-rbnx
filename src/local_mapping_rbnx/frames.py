# SPDX-License-Identifier: MulanPSL-2.0
"""Small TF helpers shared by the ROS node and unit tests."""
from __future__ import annotations

import math

import numpy as np


def stamp_seconds(message, field: str = "header") -> float:
    header = getattr(message, field, message)
    stamp = header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("TF quaternion is invalid")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, translation, rotation) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    matrix = quaternion_matrix(float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
    out = pts @ matrix.T
    out += np.asarray([translation.x, translation.y, translation.z], dtype=np.float64)
    return out


def transform_origin(transform_message) -> np.ndarray:
    t = transform_message.transform.translation
    return np.asarray([t.x, t.y, t.z], dtype=np.float64)


def yaw_from_transform(transform_message) -> float:
    q = transform_message.transform.rotation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def compose_transforms(parent_to_child, child_to_grandchild):
    """Compose two TransformStamped messages: parent -> grandchild."""
    p = parent_to_child.transform
    c = child_to_grandchild.transform
    r1 = quaternion_matrix(p.rotation.x, p.rotation.y, p.rotation.z, p.rotation.w)
    r2 = quaternion_matrix(c.rotation.x, c.rotation.y, c.rotation.z, c.rotation.w)
    t1 = np.asarray([p.translation.x, p.translation.y, p.translation.z], dtype=np.float64)
    t2 = np.asarray([c.translation.x, c.translation.y, c.translation.z], dtype=np.float64)
    rotation = r1 @ r2
    translation = t1 + r1 @ t2
    return translation, rotation


def matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax([rotation[0, 0], rotation[1, 1], rotation[2, 2]]))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return float(x), float(y), float(z), float(w)