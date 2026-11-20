# SPDX-License-Identifier: MulanPSL-2.0
"""The complete input surface of the local mapper.

Only the contracts in CONTRACTS can be resolved. There is deliberately no
path from configuration to any other Atlas capability.
"""
from __future__ import annotations

from dataclasses import dataclass

from local_mapping_rbnx.params import MapConfig

CONTRACTS: dict[str, str] = {
    "scan_rings": "robonix/primitive/lidar/lidar_rings",
    "scan_low": "robonix/primitive/lidar/lidar_low",
    "scan_high": "robonix/primitive/lidar/lidar_high",
    "scan": "robonix/primitive/lidar/lidar",
    "depth": "robonix/primitive/camera/depth",
    "rgb": "robonix/primitive/camera/rgb",
    "intrinsics": "robonix/primitive/camera/intrinsics",
    "extrinsics": "robonix/primitive/camera/extrinsics",
    "odom": "robonix/primitive/chassis/odom",
}

ROLE_MESSAGE = {
    "scan_rings": "PointCloud2",
    "scan_low": "LaserScan",
    "scan_high": "LaserScan",
    "scan": "LaserScan",
    "depth": "Image",
    "rgb": "Image",
    "intrinsics": "CameraInfo",
    "extrinsics": "TransformStamped",
    "odom": "Odometry",
}


@dataclass
class ResolvedSensor:
    role: str
    provider_id: str
    topic: str
    contract: str


def is_allowed_role(role: str) -> bool:
    return role in CONTRACTS


def is_allowed_frame(cfg: MapConfig, frame: str) -> bool:
    value = (frame or "").strip()
    if not value:
        return False
    return value in cfg.allowed_frames or value in (cfg.map_frame, cfg.base_frame)


def geometry_roles(cfg: MapConfig) -> tuple[str, ...]:
    """Pick the highest-fidelity bound lidar role, plus depth when bound."""
    roles: list[str] = []
    for role in ("scan_rings", "scan_low", "scan_high", "scan"):
        if role in cfg.sensor_providers:
            roles.append(role)
            break
    if "depth" in cfg.sensor_providers:
        roles.append("depth")
    return tuple(roles)