# SPDX-License-Identifier: MulanPSL-2.0
"""Local HTTP server for inspecting the rolling local map and sending nav goals."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

_DIRECT_OPENER = build_opener(ProxyHandler({}))
_VISUAL_OBJECT_POST_ROUTES = frozenset(
    {"scan", "approach", "inspect", "withdraw", "deposit", "clear"}
)

log = logging.getLogger("local_mapping_rbnx.webui")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

NAV_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"})
MAX_POST_BYTES = 16 * 1024

_NAVIGATION_RPC = {
    "navigate": (
        "robonix/service/navigation/navigate",
        "RobonixServiceNavigationNavigateStub",
        "Navigate",
        "Navigate_Request",
    ),
    "status": (
        "robonix/service/navigation/navigate/status",
        "RobonixServiceNavigationNavigateStatusStub",
        "GetNavigationStatus",
        "GetNavigationStatus_Request",
    ),
    "cancel": (
        "robonix/service/navigation/navigate/cancel",
        "RobonixServiceNavigationNavigateCancelStub",
        "CancelNavigation",
        "CancelNavigation_Request",
    ),
}


def _pb_to_dict(response: Any) -> dict[str, Any]:
    return {field.name: value for field, value in response.ListFields()}


def default_navigation_rpc(
    operation: str, payload: dict[str, Any], timeout_s: float = 10.0
) -> dict[str, Any]:
    """Resolve and invoke one standard Navigation gRPC capability through Atlas."""
    import grpc
    import navigation_pb2  # type: ignore
    import robonix_contracts_pb2_grpc as contracts_grpc  # type: ignore
    from robonix_api import ATLAS, Transport

    contract_id, stub_name, method_name, request_name = _NAVIGATION_RPC[operation]
    channel_ref = None
    try:
        capabilities = ATLAS.find_capability(
            contract_id=contract_id,
            transport=Transport.GRPC,
        )
        if not capabilities:
            return {"ok": False, "detail": f"no provider for {contract_id}"}
        provider = capabilities[0]
        channel_ref = ATLAS.connect_capability(
            consumer_id=os.environ.get("ROBONIX_CAPABILITY_ID", "local_mapping"),
            provider_id=provider.provider_id,
            contract_id=contract_id,
            transport=Transport.GRPC,
        )
        endpoint = (channel_ref.endpoint or "").strip()
        if not endpoint:
            return {
                "ok": False,
                "detail": f"{contract_id} resolved to an empty endpoint",
            }
        with grpc.insecure_channel(endpoint) as grpc_channel:
            stub = getattr(contracts_grpc, stub_name)(grpc_channel)
            request = getattr(navigation_pb2, request_name)(**payload)
            response = getattr(stub, method_name)(request, timeout=timeout_s)
        return _pb_to_dict(response)
    except grpc.RpcError as exc:
        return {
            "ok": False,
            "detail": (
                f"{contract_id} rpc failed: "
                f"{exc.code().name}: {exc.details()}"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("webui navigation rpc %s failed", contract_id)
        return {"ok": False, "detail": str(exc)}
    finally:
        if channel_ref is not None:
            try:
                channel_ref.close()
            except Exception:  # noqa: BLE001
                pass


class WebNavigationController:
    """Manages click-to-navigate goals via standard robonix/service/navigation/* RPCs."""

    def __init__(
        self,
        frame_id: str,
        cell_validator: Callable[[float, float], tuple[bool, str]],
        robot_pose_provider: Callable[[], tuple[float, float, float]],
        rpc_fn: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
        status_poll_interval_s: float = 0.4,
    ) -> None:
        self._frame_id = frame_id
        self._cell_validator = cell_validator
        self._robot_pose_provider = robot_pose_provider
        self._rpc = rpc_fn or default_navigation_rpc
        self._status_poll_interval_s = float(status_poll_interval_s)
        self._lock = threading.Lock()
        self._last_poll = 0.0
        self._state: dict[str, Any] = {
            "run_id": "",
            "state": "IDLE",
            "detail": "no navigation goal",
            "frame_id": frame_id,
            "goal_x": None,
            "goal_y": None,
            "goal_yaw": None,
        }

    def snapshot(self, *, refresh: bool = False) -> dict[str, Any]:
        if refresh:
            self._maybe_refresh()
        with self._lock:
            return dict(self._state)

    def _maybe_refresh(self) -> None:
        now = time.monotonic()
        with self._lock:
            run_id = str(self._state.get("run_id") or "")
            state = str(self._state.get("state") or "IDLE").upper()
            if not run_id or state in NAV_TERMINAL:
                return
            if now - self._last_poll < self._status_poll_interval_s:
                return
            self._last_poll = now
        response = self._rpc("status", {"run_id": run_id}, 5.0)
        if response.get("ok") is False:
            return
        if not bool(response.get("known", False)):
            with self._lock:
                if self._state.get("run_id") == run_id:
                    self._state["state"] = "FAILED"
                    self._state["detail"] = str(
                        response.get("detail") or "navigation lost the run"
                    )
            return
        new_state = str(response.get("state") or "RUNNING").upper()
        detail = str(response.get("detail") or new_state.lower())
        with self._lock:
            if self._state.get("run_id") == run_id:
                self._state["state"] = new_state
                self._state["detail"] = detail

    def navigate(
        self, x: float, y: float, yaw: float | None = None
    ) -> dict[str, Any]:
        if yaw is None:
            rx, ry, ryaw = self._robot_pose_provider()
            dx, dy = x - rx, y - ry
            yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-3 else ryaw
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return {"ok": False, "accepted": False, "detail": "goal must be finite"}
        current = self.snapshot(refresh=True)
        if current["run_id"] and current["state"] not in NAV_TERMINAL:
            return {
                "ok": False,
                "accepted": False,
                "detail": (
                    f"navigation is still {str(current['state']).lower()}; "
                    "cancel it first"
                ),
            }
        allowed, cell_detail = self._cell_validator(x, y)
        if not allowed:
            return {"ok": False, "accepted": False, "detail": cell_detail}

        half_yaw = yaw / 2.0
        response = self._rpc(
            "navigate",
            {
                "goal": {
                    "header": {
                        "stamp": {"sec": 0, "nanosec": 0},
                        "frame_id": self._frame_id,
                    },
                    "pose": {
                        "position": {"x": float(x), "y": float(y), "z": 0.0},
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": math.sin(half_yaw),
                            "w": math.cos(half_yaw),
                        },
                    },
                }
            },
            10.0,
        )
        accepted = bool(response.get("accepted", False))
        run_id = str(response.get("run_id") or "")
        detail = str(response.get("detail") or "")
        if not accepted or not run_id:
            return {
                **response,
                "ok": False,
                "accepted": False,
                "detail": detail or "navigation rejected the goal",
            }
        with self._lock:
            self._last_poll = 0.0
            self._state.update(
                {
                    "run_id": run_id,
                    "state": "RUNNING",
                    "detail": detail or cell_detail,
                    "frame_id": self._frame_id,
                    "goal_x": round(float(x), 3),
                    "goal_y": round(float(y), 3),
                    "goal_yaw": round(float(yaw), 4),
                }
            )
            nav_state = dict(self._state)
        log.info("webui navigation started %s -> (%.2f, %.2f, %.1f deg)", run_id, x, y, math.degrees(yaw))
        return {"ok": True, "accepted": True, "navigation": nav_state, **response}

    def cancel(self) -> dict[str, Any]:
        current = self.snapshot(refresh=False)
        run_id = str(current.get("run_id") or "")
        if not run_id:
            return {"ok": False, "accepted": False, "detail": "no active navigation run"}
        if current.get("state") in NAV_TERMINAL:
            return {
                "ok": True,
                "accepted": True,
                "detail": f"navigation is already {str(current['state']).lower()}",
                "navigation": current,
            }
        response = self._rpc("cancel", {"run_id": run_id}, 10.0)
        accepted = bool(response.get("accepted", False))
        detail = str(response.get("detail") or "")
        if not accepted:
            return {
                **response,
                "ok": False,
                "accepted": False,
                "detail": detail or "navigation cancel rejected",
            }
        with self._lock:
            if self._state.get("run_id") == run_id:
                self._state["state"] = "CANCELED"
                self._state["detail"] = detail or "canceled"
            nav_state = dict(self._state)
        log.info("webui navigation canceled %s", run_id)
        return {"ok": True, "accepted": True, "navigation": nav_state, **response}


class LocalMapWebUI:
    """Lightweight local HTTP server for the rolling local map and goal control."""

    def __init__(
        self,
        host: str,
        port: int,
        snapshot_provider: Callable[[], dict[str, Any]],
        health_provider: Callable[[], dict[str, Any]],
        nav_controller: WebNavigationController | None = None,
        reset_fn: Callable[[], dict[str, Any]] | None = None,
        visual_object_url: str = "",
    ) -> None:
        self._host = host
        self._port = int(port)
        self._snapshot_provider = snapshot_provider
        self._health_provider = health_provider
        self._nav_controller = nav_controller
        self._reset_fn = reset_fn
        self._visual_object_url = (visual_object_url or "").strip().rstrip("/")
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return
        html_bytes = INDEX_HTML.read_bytes()
        snapshot_provider = self._snapshot_provider
        health_provider = self._health_provider
        nav_controller = self._nav_controller
        reset_fn = self._reset_fn
        vo_url = self._visual_object_url

        def _proxy_vo_get(subpath: str, timeout_s: float = 2.0) -> tuple[int, str, bytes]:
            if not vo_url:
                body = json.dumps(
                    {"ok": False, "available": False, "detail": "visual_object_url 未配置"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                return int(HTTPStatus.OK), "application/json; charset=utf-8", body
            url = f"{vo_url}/api/{subpath}"
            req = Request(url, method="GET")
            try:
                with _DIRECT_OPENER.open(req, timeout=timeout_s) as resp:
                    ctype = resp.headers.get("Content-Type") or "application/octet-stream"
                    return int(resp.status), ctype, resp.read()
            except HTTPError as exc:
                ctype = exc.headers.get("Content-Type") if exc.headers else "application/json; charset=utf-8"
                return int(exc.code), ctype or "application/json; charset=utf-8", exc.read()
            except (URLError, OSError, TimeoutError) as exc:
                if subpath == "preview.jpg":
                    return int(HTTPStatus.NO_CONTENT), "application/json; charset=utf-8", b""
                body = json.dumps(
                    {"ok": False, "available": False, "detail": f"visual_object 服务未连接: {exc}"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                return int(HTTPStatus.OK), "application/json; charset=utf-8", body

        def _proxy_vo_post(subpath: str, raw_body: bytes, timeout_s: float = 60.0) -> tuple[int, str, bytes]:
            if not vo_url:
                body = json.dumps(
                    {"ok": False, "detail": "visual_object_url 未配置"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                return int(HTTPStatus.SERVICE_UNAVAILABLE), "application/json; charset=utf-8", body
            url = f"{vo_url}/api/{subpath}"
            req = Request(
                url,
                data=raw_body or b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with _DIRECT_OPENER.open(req, timeout=timeout_s) as resp:
                    ctype = resp.headers.get("Content-Type") or "application/json; charset=utf-8"
                    return int(resp.status), ctype, resp.read()
            except HTTPError as exc:
                ctype = exc.headers.get("Content-Type") if exc.headers else "application/json; charset=utf-8"
                return int(exc.code), ctype or "application/json; charset=utf-8", exc.read()
            except (URLError, OSError, TimeoutError) as exc:
                body = json.dumps(
                    {"ok": False, "detail": f"visual_object 请求失败: {exc}"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                return int(HTTPStatus.BAD_GATEWAY), "application/json; charset=utf-8", body

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_bytes(
                self, status: HTTPStatus, content_type: str, body: bytes
            ) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self._send_bytes(status, "application/json; charset=utf-8", body)

            def _read_json_body(self) -> dict[str, Any]:
                length_raw = self.headers.get("Content-Length", "0")
                try:
                    length = int(length_raw)
                except ValueError as exc:
                    raise ValueError("invalid Content-Length") from exc
                if length < 0 or length > MAX_POST_BYTES:
                    raise ValueError("request body too large")
                if length == 0:
                    return {}
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("JSON body must be an object")
                return data

            def do_HEAD(self) -> None:  # noqa: N802
                self.do_GET()

            def do_GET(self) -> None:  # noqa: N802
                route = urlsplit(self.path).path.strip("/")
                if route in ("", "index.html"):
                    self._send_bytes(
                        HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML.read_bytes()
                    )
                    return
                if route == "api/visual_object/state":
                    code, ctype, body = _proxy_vo_get("state", timeout_s=2.0)
                    self._send_bytes(HTTPStatus(code), ctype, body)
                    return
                if route == "api/visual_object/preview.jpg":
                    code, ctype, body = _proxy_vo_get("preview.jpg", timeout_s=2.0)
                    self._send_bytes(HTTPStatus(code), ctype, body)
                    return
                if route == "api/state":
                    try:
                        payload = snapshot_provider()
                        if nav_controller is not None:
                            payload["navigation"] = nav_controller.snapshot(refresh=True)
                    except Exception as exc:  # noqa: BLE001
                        self._send_json(
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"ok": False, "error": str(exc)},
                        )
                        return
                    self._send_json(HTTPStatus.OK, payload)
                    return
                if route == "api/health":
                    try:
                        payload = health_provider()
                    except Exception as exc:  # noqa: BLE001
                        self._send_json(
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"ok": False, "error": str(exc)},
                        )
                        return
                    self._send_json(HTTPStatus.OK, payload)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                route = urlsplit(self.path).path.strip("/")
                if route == "api/navigate":
                    if nav_controller is None:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {"ok": False, "detail": "navigation control not enabled"},
                        )
                        return
                    try:
                        body = self._read_json_body()
                        x = float(body["x"])
                        y = float(body["y"])
                        yaw_raw = body.get("yaw")
                        yaw = None if yaw_raw is None else float(yaw_raw)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"ok": False, "detail": f"invalid goal payload: {exc}"},
                        )
                        return
                    result = nav_controller.navigate(x, y, yaw)
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    self._send_json(status, result)
                    return
                if route == "api/navigate/cancel":
                    if nav_controller is None:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {"ok": False, "detail": "navigation control not enabled"},
                        )
                        return
                    result = nav_controller.cancel()
                    status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                    self._send_json(status, result)
                    return
                if route == "api/map/reset":
                    if reset_fn is None:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {"ok": False, "detail": "map reset not enabled"},
                        )
                        return
                    try:
                        result = reset_fn()
                    except Exception as exc:  # noqa: BLE001
                        self._send_json(
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"ok": False, "detail": str(exc)},
                        )
                        return
                    self._send_json(HTTPStatus.OK, result)
                    return
                if route.startswith("api/visual_object/"):
                    action = route[len("api/visual_object/") :]
                    if action not in _VISUAL_OBJECT_POST_ROUTES:
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "detail": "unknown visual_object action"})
                        return
                    try:
                        body = self._read_json_body()
                    except (ValueError, json.JSONDecodeError) as exc:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "detail": f"invalid payload: {exc}"})
                        return
                    code, ctype, raw = _proxy_vo_post(
                        action,
                        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                        timeout_s=60.0,
                    )
                    self._send_bytes(HTTPStatus(code), ctype, raw)
                    return
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "unsupported endpoint"},
                )

            def _reject_write(self) -> None:
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "method not allowed"},
                )

            do_PUT = _reject_write
            do_PATCH = _reject_write
            do_DELETE = _reject_write

        class _ReusableHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _ReusableHTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.25},
            name="local-map-webui",
            daemon=True,
        )
        self._thread.start()
        log.info("local map web UI listening on %s", self.url)

    def stop(self) -> None:
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                server.server_close()
            except Exception:  # noqa: BLE001
                pass
        if thread is not None:
            thread.join(timeout=2.0)
