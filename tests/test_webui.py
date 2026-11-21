# SPDX-License-Identifier: MulanPSL-2.0
from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from local_mapping_rbnx import params
from local_mapping_rbnx.node import LocalMapNode
from local_mapping_rbnx.webui import LocalMapWebUI, WebNavigationController

ROOT = Path(__file__).resolve().parents[1]


def make_node() -> LocalMapNode:
    cfg = params.from_config(
        {
            "sensor_providers": {"scan": "lidar0", "odom": "chassis0"},
            "grid_size_m": 16,
            "publish_size_m": 8,
            "resolution_m": 0.5,
            "webui_enabled": True,
            "webui_host": "127.0.0.1",
            "webui_port": 0,
        },
        ROOT,
    )
    node = LocalMapNode(cfg, {"scan": "/robonix/local_mapping/test_scan", "odom": "/robonix/local_mapping/test_odom"})
    node.grid.integrate_lidar(
        (0.0, 0.0),
        np.array([[2.0, 0.0, 0.5], [2.0, 0.5, 0.6]], dtype=np.float64),
        stamp=1.0,
        ground_reference_z=0.0,
    )
    return node


def test_webui_serves_html_state_health_and_rejects_writes():
    node = make_node()
    webui = LocalMapWebUI(
        "127.0.0.1",
        0,
        node.webui_snapshot,
        node.health_payload,
    )
    webui.start()
    try:
        assert webui.port > 0
        with urlopen(f"{webui.url}/", timeout=3) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "Local Mapping" in body
            assert "/api/state" in body
            assert "/api/navigate" in body

        with urlopen(f"{webui.url}/api/health", timeout=3) as resp:
            assert resp.status == 200
            health = json.loads(resp.read().decode("utf-8"))
            assert health["frame"] == "minecraft_odom"
            assert health["semantics_enabled"] is False
            assert health["grid"]["occupied_cells"] >= 1

        with urlopen(f"{webui.url}/api/state", timeout=3) as resp:
            assert resp.status == 200
            state = json.loads(resp.read().decode("utf-8"))
            win = state["window"]
            assert win["width"] == 16
            assert win["height"] == 16
            assert win["resolution"] == pytest.approx(0.5)
            cost = np.frombuffer(base64.b64decode(win["cost_b64"]), dtype=np.int8)
            occ = np.frombuffer(base64.b64decode(win["occupancy_b64"]), dtype=np.int8)
            assert cost.size == 16 * 16
            assert occ.size == 16 * 16
            assert np.any(cost == 100)
            assert win["obstacle_count"] >= 1
            assert len(win["obstacles"]) == win["obstacle_count"]
            assert state["config"]["sensor_roles"] == ["odom", "scan"]

        with pytest.raises(HTTPError) as exc_info:
            urlopen(Request(f"{webui.url}/api/state", data=b"{}", method="POST"), timeout=3)
        assert exc_info.value.code == 405
    finally:
        webui.stop()


def test_webui_click_to_navigate_and_cancel():
    node = make_node()
    rpc_calls = []

    def fake_rpc(op, payload, timeout_s=10.0):
        rpc_calls.append((op, payload))
        if op == "navigate":
            return {"accepted": True, "run_id": "run-123", "detail": "queued"}
        if op == "status":
            return {"known": True, "state": "RUNNING", "detail": "moving"}
        if op == "cancel":
            return {"accepted": True, "detail": "canceled"}
        return {"ok": False}

    nav = WebNavigationController(
        frame_id=node.cfg.map_frame,
        cell_validator=node.validate_goal_cell,
        robot_pose_provider=node._robot_pose,
        rpc_fn=fake_rpc,
        status_poll_interval_s=0.0,
    )
    webui = LocalMapWebUI(
        "127.0.0.1",
        0,
        node.webui_snapshot,
        node.health_payload,
        nav_controller=nav,
        reset_fn=node.reset_map,
    )
    webui.start()
    try:
        # Lethal cell at (2.0, 0.0) must be rejected (409).
        with pytest.raises(HTTPError) as exc_info:
            urlopen(
                Request(
                    f"{webui.url}/api/navigate",
                    data=json.dumps({"x": 2.0, "y": 0.0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=3,
            )
        assert exc_info.value.code == 409

        # Free cell at (1.0, 0.0) succeeds and sends frame_id=minecraft_odom.
        with urlopen(
            Request(
                f"{webui.url}/api/navigate",
                data=json.dumps({"x": 1.0, "y": 0.0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=3,
        ) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["ok"] is True
            assert res["navigation"]["run_id"] == "run-123"
            assert res["navigation"]["state"] == "RUNNING"

        assert rpc_calls[0][0] == "navigate"
        assert rpc_calls[0][1]["goal"]["header"]["frame_id"] == "minecraft_odom"

        # State includes navigation status.
        with urlopen(f"{webui.url}/api/state", timeout=3) as resp:
            state = json.loads(resp.read().decode("utf-8"))
            assert state["navigation"]["run_id"] == "run-123"
            assert state["navigation"]["state"] == "RUNNING"

        # Cancel succeeds.
        with urlopen(
            Request(f"{webui.url}/api/navigate/cancel", data=b"{}", method="POST"),
            timeout=3,
        ) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["ok"] is True
            assert res["navigation"]["state"] == "CANCELED"

        # Map reset clears occupied cells.
        assert node.grid.stats()["occupied_cells"] >= 1
        with urlopen(
            Request(f"{webui.url}/api/map/reset", data=b"{}", method="POST"),
            timeout=3,
        ) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["ok"] is True
        assert node.grid.stats()["occupied_cells"] == 0
    finally:
        webui.stop()


def test_transform_points_falls_back_to_latest_tf_within_max_age():
    from types import SimpleNamespace

    node = make_node()

    def make_tf(sec: int, nanosec: int):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=1.0, y=2.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

    def fake_lookup(target, source, stamp, *, timeout=None):
        assert timeout == 0.0
        if stamp is not None:
            raise RuntimeError("extrapolation into the future")
        return make_tf(100, 200_000_000)

    node._lookup = fake_lookup
    pts = np.array([[2.0, 0.0, 0.5]], dtype=np.float64)

    # 36 ms skew is within default max_tf_age_s (0.5 s) -> accepted.
    origin, transformed = node._transform_points(
        pts, "minecraft_view_base", SimpleNamespace(sec=100, nanosec=236_000_000)
    )
    assert origin is not None
    assert np.allclose(transformed[0], [3.0, 2.0, 0.5])

    # 2.0 s skew exceeds max_tf_age_s (0.5 s) -> rejected.
    origin_stale, _ = node._transform_points(
        pts, "minecraft_view_base", SimpleNamespace(sec=102, nanosec=200_000_000)
    )
    assert origin_stale is None
