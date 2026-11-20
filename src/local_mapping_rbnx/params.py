# SPDX-License-Identifier: MulanPSL-2.0
"""Strict validation of the deployment config delivered by CMD_INIT.

This module is deliberately free of ROS imports so the geometry and contract
tests can run without a sourced ROS environment.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TOPIC_RE = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*$")
FRAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_/]*$")

SENSOR_ROLES = (
    "scan_rings",
    "scan_low",
    "scan_high",
    "scan",
    "depth",
    "rgb",
    "intrinsics",
    "extrinsics",
    "odom",
)
GEOMETRY_ROLES = ("scan_rings", "scan_low", "scan_high", "scan", "depth")

TOP_LEVEL_KEYS = {
    "sensor_providers",
    "map_frame",
    "base_frame",
    "odom_frame",
    "base_height_m",
    "use_sim_time",
    "grid_size_m",
    "publish_size_m",
    "resolution_m",
    "update_rate_hz",
    "publish_rate_hz",
    "tf_timeout_s",
    "max_tf_age_s",
    "depth_stride",
    "depth_min_range_m",
    "depth_max_range_m",
    "depth_clearing",
    "lidar_min_range_m",
    "lidar_max_range_m",
    "ground_percentile",
    "step_height_m",
    "slope_max_deg",
    "obstacle_min_height_m",
    "obstacle_max_height_m",
    "log_odds_free",
    "log_odds_occupied",
    "log_odds_min",
    "log_odds_max",
    "decay_seconds",
    "semantic_model_path",
    "semantic_width",
    "semantic_height",
    "semantic_rate_hz",
    "semantic_conf_threshold",
    "hazard_conf_threshold",
    "semantic_ema_frames",
    "rgbd_max_skew_s",
    "cost_grid_topic",
    "occupancy_topic",
    "obstacles_topic",
    "health_topic",
    "health_rate_hz",
    "allowed_input_frames",
}


class ConfigError(RuntimeError):
    """A deployment config value is invalid."""


@dataclass(frozen=True)
class MapConfig:
    sensor_providers: dict[str, str]
    map_frame: str = "minecraft_odom"
    base_frame: str = "minecraft_view_base"
    odom_frame: str = "minecraft_odom"
    base_height_m: float = 1.62
    use_sim_time: bool = True
    grid_size_m: float = 64.0
    publish_size_m: float = 32.0
    resolution_m: float = 0.2
    update_rate_hz: float = 10.0
    publish_rate_hz: float = 10.0
    tf_timeout_s: float = 0.25
    max_tf_age_s: float = 0.5
    depth_stride: int = 4
    depth_min_range_m: float = 0.25
    depth_max_range_m: float = 12.0
    depth_clearing: bool = False
    lidar_min_range_m: float = 0.3
    lidar_max_range_m: float = 32.0
    ground_percentile: float = 10.0
    step_height_m: float = 0.55
    slope_max_deg: float = 35.0
    obstacle_min_height_m: float = 0.05
    obstacle_max_height_m: float = 2.0
    log_odds_free: float = 0.4
    log_odds_occupied: float = 0.85
    log_odds_min: float = -5.0
    log_odds_max: float = 5.0
    decay_seconds: float = 8.0
    semantic_model_path: str = ""
    semantic_width: int = 320
    semantic_height: int = 192
    semantic_rate_hz: float = 5.0
    semantic_conf_threshold: float = 0.55
    hazard_conf_threshold: float = 0.75
    semantic_ema_frames: int = 3
    rgbd_max_skew_s: float = 0.08
    cost_grid_topic: str = "/robonix/local_mapping/cost_grid"
    occupancy_topic: str = "/robonix/local_mapping/occupancy"
    obstacles_topic: str = "/robonix/local_mapping/obstacles"
    health_topic: str = "/robonix/local_mapping/health"
    health_rate_hz: float = 1.0
    allowed_frames: frozenset[str] = field(default_factory=frozenset)

    @property
    def cells(self) -> int:
        return int(round(self.grid_size_m / self.resolution_m))

    @property
    def publish_cells(self) -> int:
        return int(round(self.publish_size_m / self.resolution_m))

    @property
    def semantic_enabled(self) -> bool:
        return bool(self.semantic_model_path) and (
            "rgb" in self.sensor_providers and "depth" in self.sensor_providers
        )


def _number(raw: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(raw, bool):
        raise ConfigError(f"{name} must be a number, not a boolean")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value < lo or value > hi:
        raise ConfigError(f"{name}={raw!r} must be finite and within [{lo}, {hi}]")
    return value


def _integer(raw: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(raw, bool):
        raise ConfigError(f"{name} must be an integer, not a boolean")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise ConfigError(f"{name}={raw!r} must be an exact integer")
    if value < lo or value > hi:
        raise ConfigError(f"{name}={value} must be within [{lo}, {hi}]")
    return value


def _boolean(raw: Any, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    raise ConfigError(f"{name} must be a boolean")


def _string(raw: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(raw, str):
        raise ConfigError(f"{name} must be a string")
    value = raw.strip()
    if not value and not allow_empty:
        raise ConfigError(f"{name} must be non-empty")
    return value


def validate_topic(raw: Any, name: str) -> str:
    value = _string(raw, name)
    if not TOPIC_RE.fullmatch(value):
        raise ConfigError(f"{name}={value!r} must be an absolute ROS topic name")
    return value


def validate_frame(raw: Any, name: str) -> str:
    value = _string(raw, name)
    if not FRAME_RE.fullmatch(value):
        raise ConfigError(f"{name}={value!r} is not a valid TF frame")
    return value


def resolve_model_path(raw: Any, package_root: Path) -> str:
    value = _string(raw, "semantic_model_path", allow_empty=True)
    if not value:
        return ""
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = package_root / path
    return str(path)


def from_config(config: dict[str, Any], package_root: Path) -> MapConfig:
    if not isinstance(config, dict):
        raise ConfigError("config must be a mapping")
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown config key(s): {sorted(unknown)}")

    raw_providers = config.get("sensor_providers")
    if not isinstance(raw_providers, dict) or not raw_providers:
        raise ConfigError("sensor_providers must be a non-empty mapping")
    unknown_roles = set(raw_providers) - set(SENSOR_ROLES)
    if unknown_roles:
        raise ConfigError(
            f"unknown sensor role(s) {sorted(unknown_roles)}; "
            f"options: {list(SENSOR_ROLES)}"
        )
    providers: dict[str, str] = {}
    for role, raw in raw_providers.items():
        providers[role] = _string(raw, f"sensor_providers.{role}")
    if "odom" not in providers:
        raise ConfigError("sensor_providers.odom is required")
    if not any(role in providers for role in GEOMETRY_ROLES):
        raise ConfigError(
            "at least one of scan_rings, scan_low, scan_high, scan or depth "
            "must be bound in sensor_providers"
        )
    if "depth" in providers and "intrinsics" not in providers:
        raise ConfigError("sensor_providers.depth requires sensor_providers.intrinsics")

    grid_size = _number(config.get("grid_size_m", 64.0), "grid_size_m", 16, 128)
    publish_size = _number(config.get("publish_size_m", 32.0), "publish_size_m", 8, 64)
    if publish_size > grid_size:
        raise ConfigError("publish_size_m must not exceed grid_size_m")
    resolution = _number(config.get("resolution_m", 0.2), "resolution_m", 0.05, 0.5)
    if int(round(grid_size / resolution)) < 8:
        raise ConfigError("grid_size_m / resolution_m must yield at least 8 cells")
    if int(round(publish_size / resolution)) < 4:
        raise ConfigError("publish_size_m / resolution_m must yield at least 4 cells")

    log_free = _number(config.get("log_odds_free", 0.4), "log_odds_free", 0.05, 2)
    log_occ = _number(config.get("log_odds_occupied", 0.85), "log_odds_occupied", 0.05, 3)
    log_min = _number(config.get("log_odds_min", -5.0), "log_odds_min", -50, 50)
    log_max = _number(config.get("log_odds_max", 5.0), "log_odds_max", -50, 50)
    if log_min >= log_max:
        raise ConfigError("log_odds_min must be less than log_odds_max")

    obstacle_min = _number(
        config.get("obstacle_min_height_m", 0.05), "obstacle_min_height_m", 0, 1
    )
    obstacle_max = _number(
        config.get("obstacle_max_height_m", 2.0), "obstacle_max_height_m", 0.2, 6
    )
    if obstacle_min >= obstacle_max:
        raise ConfigError("obstacle_min_height_m must be less than obstacle_max_height_m")

    depth_min = _number(config.get("depth_min_range_m", 0.25), "depth_min_range_m", 0.05, 20)
    depth_max = _number(config.get("depth_max_range_m", 12.0), "depth_max_range_m", 1, 64)
    if depth_min >= depth_max:
        raise ConfigError("depth_min_range_m must be less than depth_max_range_m")
    lidar_min = _number(config.get("lidar_min_range_m", 0.3), "lidar_min_range_m", 0, 20)
    lidar_max = _number(config.get("lidar_max_range_m", 32.0), "lidar_max_range_m", 2, 64)
    if lidar_min >= lidar_max:
        raise ConfigError("lidar_min_range_m must be less than lidar_max_range_m")

    map_frame = validate_frame(config.get("map_frame", "minecraft_odom"), "map_frame")
    base_frame = validate_frame(config.get("base_frame", "minecraft_view_base"), "base_frame")
    odom_frame = validate_frame(config.get("odom_frame", "minecraft_odom"), "odom_frame")
    if map_frame != odom_frame:
        raise ConfigError("map_frame must equal odom_frame for a rolling local map")

    allowed_frames = {
        map_frame,
        base_frame,
        odom_frame,
        "minecraft_camera_optical_frame",
        "minecraft_lidar_low",
        "minecraft_lidar_high",
    }
    raw_frames = config.get("allowed_input_frames", [])
    if not isinstance(raw_frames, list):
        raise ConfigError("allowed_input_frames must be a list of TF frame names")
    for index, raw_frame in enumerate(raw_frames):
        allowed_frames.add(validate_frame(raw_frame, f"allowed_input_frames[{index}]"))

    semantic_width = _integer(config.get("semantic_width", 320), "semantic_width", 64, 1280)
    semantic_height = _integer(config.get("semantic_height", 192), "semantic_height", 64, 720)

    return MapConfig(
        sensor_providers=providers,
        map_frame=map_frame,
        base_frame=base_frame,
        odom_frame=odom_frame,
        base_height_m=_number(config.get("base_height_m", 1.62), "base_height_m", 0, 5),
        use_sim_time=_boolean(config.get("use_sim_time", True), "use_sim_time"),
        grid_size_m=grid_size,
        publish_size_m=publish_size,
        resolution_m=resolution,
        update_rate_hz=_number(config.get("update_rate_hz", 10.0), "update_rate_hz", 1, 30),
        publish_rate_hz=_number(config.get("publish_rate_hz", 10.0), "publish_rate_hz", 1, 30),
        tf_timeout_s=_number(config.get("tf_timeout_s", 0.25), "tf_timeout_s", 0.05, 2),
        max_tf_age_s=_number(config.get("max_tf_age_s", 0.5), "max_tf_age_s", 0.05, 2),
        depth_stride=_integer(config.get("depth_stride", 4), "depth_stride", 1, 16),
        depth_min_range_m=depth_min,
        depth_max_range_m=depth_max,
        depth_clearing=_boolean(config.get("depth_clearing", False), "depth_clearing"),
        lidar_min_range_m=lidar_min,
        lidar_max_range_m=lidar_max,
        ground_percentile=_number(config.get("ground_percentile", 10.0), "ground_percentile", 1, 50),
        step_height_m=_number(config.get("step_height_m", 0.55), "step_height_m", 0.05, 2),
        slope_max_deg=_number(config.get("slope_max_deg", 35.0), "slope_max_deg", 5, 60),
        obstacle_min_height_m=obstacle_min,
        obstacle_max_height_m=obstacle_max,
        log_odds_free=log_free,
        log_odds_occupied=log_occ,
        log_odds_min=log_min,
        log_odds_max=log_max,
        decay_seconds=_number(config.get("decay_seconds", 8.0), "decay_seconds", 0, 120),
        semantic_model_path=resolve_model_path(
            config.get("semantic_model_path", ""), package_root
        ),
        semantic_width=semantic_width,
        semantic_height=semantic_height,
        semantic_rate_hz=_number(config.get("semantic_rate_hz", 5.0), "semantic_rate_hz", 0.5, 15),
        semantic_conf_threshold=_number(
            config.get("semantic_conf_threshold", 0.55), "semantic_conf_threshold", 0, 1
        ),
        hazard_conf_threshold=_number(
            config.get("hazard_conf_threshold", 0.75), "hazard_conf_threshold", 0, 1
        ),
        semantic_ema_frames=_integer(
            config.get("semantic_ema_frames", 3), "semantic_ema_frames", 1, 20
        ),
        rgbd_max_skew_s=_number(config.get("rgbd_max_skew_s", 0.08), "rgbd_max_skew_s", 0, 1),
        cost_grid_topic=validate_topic(
            config.get("cost_grid_topic", "/robonix/local_mapping/cost_grid"),
            "cost_grid_topic",
        ),
        occupancy_topic=validate_topic(
            config.get("occupancy_topic", "/robonix/local_mapping/occupancy"),
            "occupancy_topic",
        ),
        obstacles_topic=validate_topic(
            config.get("obstacles_topic", "/robonix/local_mapping/obstacles"),
            "obstacles_topic",
        ),
        health_topic=validate_topic(
            config.get("health_topic", "/robonix/local_mapping/health"),
            "health_topic",
        ),
        health_rate_hz=_number(config.get("health_rate_hz", 1.0), "health_rate_hz", 0.1, 5),
        allowed_frames=frozenset(allowed_frames),
    )


def log_summary(cfg: MapConfig) -> str:
    roles = ", ".join(sorted(cfg.sensor_providers))
    semantics = "on" if cfg.semantic_enabled else "off"
    return (
        f"frame={cfg.map_frame} grid={cfg.grid_size_m:g}m@{cfg.resolution_m:g}m "
        f"publish={cfg.publish_size_m:g}m sensors=[{roles}] semantics={semantics}"
    )