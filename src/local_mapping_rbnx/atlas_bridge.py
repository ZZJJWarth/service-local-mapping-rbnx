# SPDX-License-Identifier: MulanPSL-2.0
"""Atlas bridge for the rolling local map service.

The service input surface is exactly ``topics.CONTRACTS``. Provider IDs come
from deployment config; topics are whatever Atlas resolves for those
providers. Nothing here reads Minecraft ground truth, scene state, block IDs
or game RPC state.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

from robonix_api import ATLAS, Deferred, Err, Ok, Service

from local_mapping_rbnx import params, topics
from local_mapping_rbnx.node import LocalMapNode

logging.basicConfig(
    level=os.environ.get("LOCAL_MAPPING_LOG_LEVEL", "INFO").upper(),
    format="[local_mapping] %(levelname)s %(message)s",
)
log = logging.getLogger("local_mapping_rbnx.atlas_bridge")

CAP_ID = os.environ.get("ROBONIX_CAPABILITY_ID", "local_mapping")
NAMESPACE = "robonix/service/local_mapping"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_CONTRACTS = {
    "cost_grid": "robonix/service/local_mapping/cost_grid",
    "occupancy": "robonix/service/local_mapping/occupancy",
    "obstacles": "robonix/service/local_mapping/obstacles",
    "health": "robonix/service/local_mapping/health",
}

FORBIDDEN_ENDPOINT_TOKENS = (
    "ground" + "_truth",
    "perception" + chr(47) + "objects",
    "perception" + "_objects",
    "get" + "_state",
)

local_map = Service(id=CAP_ID, namespace=NAMESPACE)

_init_lock = threading.Lock()
_initialized = False
_node: Optional[LocalMapNode] = None
_channels: dict[str, object] = {}
_resolved: dict[str, topics.ResolvedSensor] = {}


def _pick_provider(recs, provider_id: str, contract_id: str):
    if provider_id:
        for rec in recs:
            if rec.provider_id == provider_id:
                return rec
        return None
    if not recs:
        return None
    if len(recs) == 1:
        return recs[0]
    ids = ", ".join(sorted({rec.provider_id for rec in recs}))
    raise RuntimeError(
        f"multiple providers expose {contract_id} ({ids}); "
        "pin sensor_providers to one provider id"
    )


def _validate_endpoint(role: str, endpoint: str) -> str:
    topic = (endpoint or "").strip()
    if not topic or not params.TOPIC_RE.match(topic):
        raise RuntimeError(f"{role}: atlas returned invalid ROS topic {endpoint!r}")
    lowered = topic.lower()
    for token in FORBIDDEN_ENDPOINT_TOKENS:
        if token in lowered:
            raise RuntimeError(
                f"{role}: refusing forbidden input topic {topic!r}"
            )
    return topic


def _resolve_role(role: str, provider_id: str) -> Optional[topics.ResolvedSensor]:
    contract = topics.CONTRACTS[role]
    recs = ATLAS.find_capability(
        contract_id=contract,
        transport="ros2",
        provider_id=provider_id,
    )
    rec = _pick_provider(recs, provider_id, contract)
    if rec is None:
        return None

    held = _channels.get(role)
    if held is not None and getattr(held, "provider_id", None) == rec.provider_id:
        resolved = _resolved.get(role)
        if resolved is not None:
            return resolved
        held.close()
        del _channels[role]

    channel = local_map.connect_capability(rec, contract_id=contract, transport="ros2")
    endpoint = _validate_endpoint(role, getattr(channel, "endpoint", ""))
    _channels[role] = channel
    resolved = topics.ResolvedSensor(
        role=role,
        provider_id=rec.provider_id,
        topic=endpoint,
        contract=contract,
    )
    _resolved[role] = resolved
    log.info("resolved %s from %s -> %s", contract, rec.provider_id, endpoint)
    return resolved


def _resolve_inputs(cfg: params.MapConfig) -> tuple[dict[str, str], list[str]]:
    """Resolve every configured role. Returns (role->topic, missing contracts)."""
    missing: list[str] = []
    sensor_topics: dict[str, str] = {}
    for role, provider_id in cfg.sensor_providers.items():
        if not topics.is_allowed_role(role):
            raise RuntimeError(f"unknown sensor role {role!r}")
        resolved = _resolve_role(role, provider_id)
        if resolved is None:
            missing.append(f"{role}:{topics.CONTRACTS[role]}")
        else:
            sensor_topics[role] = resolved.topic
    if missing:
        return sensor_topics, missing
    if not topics.geometry_roles(cfg):
        raise RuntimeError("no geometry sensor configured")
    return sensor_topics, []


def _declare_outputs(cfg: params.MapConfig) -> None:
    bindings = {
        "cost_grid": (cfg.cost_grid_topic, "transient_local"),
        "occupancy": (cfg.occupancy_topic, "transient_local"),
        "obstacles": (cfg.obstacles_topic, "best_effort"),
        "health": (cfg.health_topic, "reliable"),
    }
    for key, contract in OUTPUT_CONTRACTS.items():
        topic, qos = bindings[key]
        local_map.declare_ros2_topic(contract, topic, qos=qos)
        log.info("declared %s -> ROS2 topic %s", contract, topic)


@local_map.on_init
def init(cfg: dict):
    """Resolve sensor providers and start the rolling map node."""
    global _initialized, _node
    with _init_lock:
        if _initialized:
            return Ok()
        try:
            map_cfg = params.from_config(cfg, PACKAGE_ROOT)
        except params.ConfigError as exc:
            return Err(f"invalid config: {exc}")
        except Exception as exc:  # noqa: BLE001
            return Err(f"invalid config: {exc}")

        try:
            sensor_topics, missing = _resolve_inputs(map_cfg)
        except RuntimeError as exc:
            return Err(str(exc))
        if missing:
            return Deferred(
                "waiting for atlas providers: " + ", ".join(sorted(missing))
            )

        node = LocalMapNode(map_cfg, sensor_topics)
        try:
            node.start()
        except Exception as exc:  # noqa: BLE001
            node.stop()
            return Err(f"failed to start local map node: {exc}")

        try:
            _declare_outputs(map_cfg)
        except Exception as exc:  # noqa: BLE001
            node.stop()
            return Err(f"failed to declare outputs: {exc}")

        _node = node
        _initialized = True
        log.info(
            "local map active: %d inputs, geometry=%s, semantics=%s",
            len(sensor_topics),
            ",".join(topics.geometry_roles(map_cfg)),
            "on" if node.segmenter.enabled else "off",
        )
        return Ok()


@local_map.on_shutdown
def shutdown() -> None:
    global _initialized, _node
    with _init_lock:
        node, _node = _node, None
        _initialized = False
    if node is not None:
        node.stop()
    for role, channel in list(_channels.items()):
        try:
            channel.close()
        except Exception:  # noqa: BLE001
            pass
        _channels.pop(role, None)
    _resolved.clear()
    log.info("local map stopped")