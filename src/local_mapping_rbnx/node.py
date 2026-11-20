# SPDX-License-Identifier: MulanPSL-2.0
"""ROS integration for the rolling local map.

Sensor messages are decoded and TF-transformed here. The grid and geometry
math stay in pure modules so they can be tested without a ROS graph.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from local_mapping_rbnx import frames, sensors, topics
from local_mapping_rbnx.grid import RollingGrid
from local_mapping_rbnx.params import MapConfig
from local_mapping_rbnx.semantics import SemanticSegmenter, depth_selected_points


class LocalMapNode:
    def __init__(self, cfg: MapConfig, sensor_topics: dict[str, str]) -> None:
        self.cfg = cfg
        self.sensor_topics = dict(sensor_topics)
        self.grid = RollingGrid(cfg)
        self.segmenter = SemanticSegmenter(cfg)
        self.lock = threading.RLock()
        self._pending: dict[str, Any] = {}
        self._pending_lock = threading.Lock()
        self._sem_lock = threading.Lock()
        self._sem_rgb: tuple[np.ndarray, float, object] | None = None
        self._sem_depth: tuple[np.ndarray, str, float, object, sensors.Intrinsics] | None = None
        self._sem_last_rgb_stamp = -1.0
        self._sem_results: deque[
            tuple[float, np.ndarray, np.ndarray, str, object]
        ] = deque(maxlen=1)
        self._sem_stop = threading.Event()
        self._sem_thread: threading.Thread | None = None
        self._intrinsics: sensors.Intrinsics | None = None
        self._latest_odom: tuple[float, float, float, float] | None = None
        self._last_grid_time = 0.0
        self._last_semantic_time = 0.0
        self._counters: dict[str, int] = {
            "frames": 0,
            "frames_dropped_frame": 0,
            "frames_dropped_tf": 0,
            "frames_invalid": 0,
            "lidar_hits": 0,
            "depth_hits": 0,
            "semantic_frames": 0,
            "semantic_skipped": 0,
            "semantic_points": 0,
        }
        self._last_error = ""
        self.node = None
        self.tf_buffer = None
        self.tf_listener = None
        self._publishers: dict[str, Any] = {}
        self._subscriptions: list[Any] = []
        self._timers: list[Any] = []
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._ros_started = False

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.parameter import Parameter

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(
            "local_mapping",
            parameter_overrides=[Parameter("use_sim_time", value=self.cfg.use_sim_time)],
        )
        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, spin_thread=False)
        self._create_publishers()
        self._create_subscriptions()
        update_period = 1.0 / self.cfg.update_rate_hz
        publish_period = 1.0 / self.cfg.publish_rate_hz
        health_period = 1.0 / self.cfg.health_rate_hz
        self._timers = [
            self.node.create_timer(update_period, self._update),
            self.node.create_timer(publish_period, self._publish),
            self.node.create_timer(health_period, self._publish_health),
        ]
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._spin, name="local-map-ros", daemon=True
        )
        self._spin_thread.start()
        self._ros_started = True
        if self.segmenter.enabled:
            self._sem_thread = threading.Thread(
                target=self._semantic_loop, name="local-map-semantics", daemon=True
            )
            self._sem_thread.start()

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception:  # noqa: BLE001
            if self._ros_started:
                self._last_error = "ros spin stopped"

    def stop(self) -> None:
        self._sem_stop.set()
        if self._sem_thread is not None:
            self._sem_thread.join(timeout=2.0)
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self._ros_started = False

    def _create_publishers(self) -> None:
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import String

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publishers["cost_grid"] = self.node.create_publisher(
            OccupancyGrid, self.cfg.cost_grid_topic, latched
        )
        self._publishers["occupancy"] = self.node.create_publisher(
            OccupancyGrid, self.cfg.occupancy_topic, latched
        )
        self._publishers["obstacles"] = self.node.create_publisher(
            PointCloud2, self.cfg.obstacles_topic, 5
        )
        self._publishers["health"] = self.node.create_publisher(
            String, self.cfg.health_topic, 5
        )

    def _create_subscriptions(self) -> None:
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
        from geometry_msgs.msg import TransformStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for role, topic in self.sensor_topics.items():
            if role == "odom":
                msg_type, qos = Odometry, qos_profile_sensor_data
            elif role in ("scan_low", "scan_high", "scan"):
                msg_type, qos = LaserScan, qos_profile_sensor_data
            elif role == "scan_rings":
                msg_type, qos = PointCloud2, qos_profile_sensor_data
            elif role in ("depth", "rgb"):
                msg_type, qos = Image, qos_profile_sensor_data
            elif role == "intrinsics":
                msg_type, qos = CameraInfo, latched
            elif role == "extrinsics":
                msg_type, qos = TransformStamped, latched
            else:
                continue
            callback = lambda msg, role=role: self._store(role, msg)
            self._subscriptions.append(self.node.create_subscription(msg_type, topic, callback, qos))

    # ── callbacks ───────────────────────────────────────────────────────────
    def _store(self, role: str, message) -> None:
        frame = (getattr(getattr(message, "header", None), "frame_id", "") or "").lstrip("/")
        if frame and not topics.is_allowed_frame(self.cfg, frame):
            with self.lock:
                self._counters["frames_dropped_frame"] += 1
            return
        with self.lock:
            self._counters["frames"] += 1
        if role == "odom":
            pose = message.pose.pose
            with self.lock:
                self._latest_odom = (
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                    frames.stamp_seconds(message),
                )
            return
        if role == "intrinsics":
            try:
                self._intrinsics = sensors.Intrinsics.from_camera_info(message)
            except ValueError as exc:
                with self.lock:
                    self._last_error = str(exc)
            return
        if role == "extrinsics":
            return
        with self._pending_lock:
            self._pending[role] = message

    def _take(self, role: str):
        with self._pending_lock:
            return self._pending.pop(role, None)

    # ── updates ─────────────────────────────────────────────────────────────
    def _update(self) -> None:
        if not self._ros_started:
            return
        now = self._now_seconds()
        with self._sem_lock:
            sem_item = self._sem_results.popleft() if self._sem_results else None
        if sem_item is not None:
            self._apply_semantic(*sem_item)
        with self.lock:
            dt = now - self._last_grid_time if self._last_grid_time else 0.0
            self._last_grid_time = now
            self.grid.decay(now, dt)
        ground_ref = self._ground_reference()
        if ground_ref is None:
            return
        lidar_roles = self._lidar_roles()
        for role in lidar_roles:
            message = self._take(role)
            if message is not None:
                self._integrate_lidar(role, message, ground_ref)
        depth_message = self._take("depth")
        if depth_message is not None:
            self._integrate_depth(depth_message, ground_ref)
        rgb_message = self._take("rgb")
        if rgb_message is not None and self.segmenter.enabled:
            self._submit_semantic(rgb_message)

    def _lidar_roles(self) -> tuple[str, ...]:
        if "scan_rings" in self.cfg.sensor_providers:
            return ("scan_rings",)
        roles = tuple(
            role for role in ("scan_low", "scan_high") if role in self.cfg.sensor_providers
        )
        if roles:
            return roles
        return ("scan",) if "scan" in self.cfg.sensor_providers else ()

    def _now_seconds(self) -> float:
        return float(self.node.get_clock().now().nanoseconds) * 1e-9

    def _lookup(self, target: str, source: str, stamp, *, timeout: float | None = None):
        from rclpy.duration import Duration
        from rclpy.time import Time

        return self.tf_buffer.lookup_transform(
            target,
            source,
            Time.from_msg(stamp) if stamp is not None else Time(),
            timeout=Duration(seconds=self.cfg.tf_timeout_s if timeout is None else timeout),
        )

    def _transform_points(self, points: np.ndarray, frame: str, stamp):
        if points.size == 0:
            return None, np.zeros((0, 3))
        try:
            transform = self._lookup(self.cfg.map_frame, frame, stamp)
        except Exception:  # noqa: BLE001
            return None, points
        translated = frames.transform_points(
            points, transform.transform.translation, transform.transform.rotation
        )
        origin = frames.transform_origin(transform)
        return origin, translated

    def _ground_reference(self) -> float | None:
        base_z = None
        try:
            transform = self._lookup(self.cfg.map_frame, self.cfg.base_frame, None, timeout=0.0)
            base_z = float(transform.transform.translation.z)
        except Exception:  # noqa: BLE001
            pass
        if base_z is None:
            with self.lock:
                odom = self._latest_odom
            if odom is None:
                return None
            base_z = odom[2]
        return base_z - self.cfg.base_height_m

    def _integrate_lidar(self, role: str, message, ground_ref: float) -> None:
        stamp = frames.stamp_seconds(message)
        frame = (message.header.frame_id or "").lstrip("/")
        try:
            if role == "scan_rings":
                points = sensors.pointcloud_xyz(
                    message.data,
                    int(message.point_step),
                    int(message.width),
                    int(message.height),
                    int(message.row_step),
                    message.fields,
                    bool(message.is_bigendian),
                )
            else:
                points = sensors.scan_endpoints(
                    message.ranges,
                    float(message.angle_min),
                    float(message.angle_increment),
                    float(message.range_min),
                    float(message.range_max),
                )
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self._counters["frames_invalid"] += 1
                self._last_error = str(exc)
            return
        origin, points = self._transform_points(points, frame, message.header.stamp)
        if origin is None:
            with self.lock:
                self._counters["frames_dropped_tf"] += 1
            return
        with self.lock:
            hits = self.grid.integrate_lidar(
                (float(origin[0]), float(origin[1])), points, stamp, ground_ref
            )
            self._counters["lidar_hits"] += hits

    def _integrate_depth(self, message, ground_ref: float) -> None:
        intr = self._intrinsics
        if intr is None:
            return
        stamp = frames.stamp_seconds(message)
        frame = (message.header.frame_id or "").lstrip("/")
        try:
            depth = sensors.depth_to_metres(
                message.data,
                message.encoding,
                int(message.height),
                int(message.width),
                int(message.step),
            )
            camera_points = sensors.backproject_depth(
                depth,
                intr,
                self.cfg.depth_stride,
                self.cfg.depth_min_range_m,
                self.cfg.depth_max_range_m,
            )
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self._counters["frames_invalid"] += 1
                self._last_error = str(exc)
            return
        origin, points = self._transform_points(camera_points, frame, message.header.stamp)
        if origin is None:
            with self.lock:
                self._counters["frames_dropped_tf"] += 1
            return
        if points.size == 0:
            return
        with self.lock:
            hits = self.grid.integrate_depth(points, stamp, ground_ref)
            self._counters["depth_hits"] += hits
        with self._sem_lock:
            self._sem_depth = (depth, frame, stamp, message.header.stamp, intr)

    # ── semantic worker ─────────────────────────────────────────────────────
    def _submit_semantic(self, rgb_message) -> None:
        if self._intrinsics is None or not self.segmenter.enabled:
            return
        try:
            rgb = sensors.decode_rgb(
                rgb_message.data,
                rgb_message.encoding,
                int(rgb_message.height),
                int(rgb_message.width),
                int(rgb_message.step),
            )
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self._counters["frames_invalid"] += 1
                self._last_error = str(exc)
            return
        with self._sem_lock:
            self._sem_rgb = (
                rgb,
                frames.stamp_seconds(rgb_message),
                rgb_message.header.stamp,
            )

    def _semantic_loop(self) -> None:
        while not self._sem_stop.is_set():
            with self._sem_lock:
                rgb_state = self._sem_rgb
                depth_state = self._sem_depth
            if rgb_state is None or depth_state is None:
                self._sem_stop.wait(0.02)
                continue
            rgb, rgb_stamp, rgb_ros_stamp = rgb_state
            depth, depth_frame, depth_stamp, depth_ros_stamp, depth_intr = depth_state
            if rgb_stamp == self._sem_last_rgb_stamp:
                self._sem_stop.wait(0.02)
                continue
            if depth.shape[:2] != rgb.shape[:2]:
                with self.lock:
                    self._counters["semantic_skipped"] += 1
                self._sem_last_rgb_stamp = rgb_stamp
                continue
            if abs(depth_stamp - rgb_stamp) > self.cfg.rgbd_max_skew_s:
                with self.lock:
                    self._counters["semantic_skipped"] += 1
                self._sem_last_rgb_stamp = rgb_stamp
                continue
            if not self.segmenter.due(self._now_seconds()):
                self._sem_stop.wait(0.02)
                continue
            segmentation = self.segmenter.infer(rgb, self._now_seconds())
            self._sem_last_rgb_stamp = rgb_stamp
            if segmentation is None:
                continue
            camera_points, costs = depth_selected_points(
                depth,
                depth_intr,
                self.cfg.depth_stride,
                self.cfg.depth_min_range_m,
                self.cfg.depth_max_range_m,
                segmentation.cost,
                segmentation.hazard_mask,
            )
            if camera_points.size == 0:
                continue
            with self._sem_lock:
                self._sem_results.append(
                    (depth_stamp, camera_points, costs, depth_frame, depth_ros_stamp)
                )
            with self.lock:
                self._counters["semantic_frames"] += 1

    def _apply_semantic(
        self, stamp: float, camera_points: np.ndarray, costs: np.ndarray, frame: str, ros_stamp
    ) -> None:
        origin, points = self._transform_points(camera_points, frame, ros_stamp)
        if origin is None or points.size == 0:
            with self.lock:
                self._counters["frames_dropped_tf"] += 1
            return
        integrated = 0
        for value in np.unique(costs):
            mask = costs == value
            with self.lock:
                integrated += self.grid.integrate_semantic(points[mask], int(value), stamp)
        with self.lock:
            self._counters["semantic_points"] += integrated

    # ── publication ─────────────────────────────────────────────────────────
    def _robot_xy(self) -> tuple[float, float]:
        try:
            transform = self._lookup(self.cfg.map_frame, self.cfg.base_frame, None, timeout=0.0)
            return (
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
            )
        except Exception:  # noqa: BLE001
            with self.lock:
                odom = self._latest_odom
            if odom is not None:
                return odom[0], odom[1]
        return 0.0, 0.0

    def _publish(self) -> None:
        from nav_msgs.msg import OccupancyGrid
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        x, y = self._robot_xy()
        with self.lock:
            window = self.grid.window((x, y))
        stamp = self.node.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self.cfg.map_frame)
        self._publishers["cost_grid"].publish(
            self._grid_message(header, window.cost, window.origin_x, window.origin_y)
        )
        self._publishers["occupancy"].publish(
            self._grid_message(header, window.occupancy, window.origin_x, window.origin_y)
        )
        points = window.obstacle_points
        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        self._publishers["obstacles"].publish(cloud)

    def _grid_message(self, header, data: np.ndarray, origin_x: float, origin_y: float):
        from nav_msgs.msg import OccupancyGrid, MapMetaData

        msg = OccupancyGrid()
        msg.header = header
        info = MapMetaData()
        info.resolution = self.cfg.resolution_m
        info.width = int(data.shape[1])
        info.height = int(data.shape[0])
        info.origin.position.x = float(origin_x)
        info.origin.position.y = float(origin_y)
        info.origin.position.z = 0.0
        info.origin.orientation.w = 1.0
        msg.info = info
        msg.data = data.flatten().tolist()
        return msg

    def _publish_health(self) -> None:
        from std_msgs.msg import String

        with self.lock:
            payload = {
                "ok": self._ros_started,
                "frame": self.cfg.map_frame,
                "semantics_enabled": self.segmenter.enabled,
                "semantics_reason": self.segmenter.reason,
                "counters": dict(self._counters),
                "grid": self.grid.stats(),
                "last_error": self._last_error,
            }
        self._publishers["health"].publish(String(data=json.dumps(payload)))