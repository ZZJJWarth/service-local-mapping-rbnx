# SPDX-License-Identifier: MulanPSL-2.0
"""Optional RGB semantic hazards.

The model is advisory. Geometry always wins: semantic costs are merged with
max() into cells that geometry has not marked lethal, and a missing model,
missing runtime, bad tensor or low confidence leaves the geometry map intact.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from local_mapping_rbnx.params import MapConfig
from local_mapping_rbnx.sensors import Intrinsics, nearest_resize, normalize_rgb


@dataclass(frozen=True)
class SemanticClass:
    name: str
    cost: int
    hazard: bool = False


# Output channel order is part of the model contract.
SEG_CLASSES: tuple[SemanticClass, ...] = (
    SemanticClass("path", 0),
    SemanticClass("grass", 15),
    SemanticClass("vegetation", 100, True),
    SemanticClass("water", 100, True),
    SemanticClass("lava", 100, True),
    SemanticClass("structure", 100, True),
    SemanticClass("unknown", 20),
)


@dataclass(frozen=True)
class Segmentation:
    cost: np.ndarray
    hazard_mask: np.ndarray
    confidence: np.ndarray
    classes: np.ndarray
    inference_ms: float


class SemanticSegmenter:
    def __init__(self, cfg: MapConfig) -> None:
        self.cfg = cfg
        self.enabled = False
        self.reason = "disabled by config"
        self._session = None
        self._input_name = ""
        self._next_allowed = 0.0
        self._ema: np.ndarray | None = None
        self._ema_weight = 1.0 / max(1, cfg.semantic_ema_frames)
        self._load()

    def _load(self) -> None:
        path = self.cfg.semantic_model_path
        if not path:
            return
        if not os.path.isfile(path):
            self.reason = f"model file not found: {path}"
            return
        try:
            import onnxruntime as ort  # type: ignore

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            self.enabled = True
            self.reason = "ready"
        except Exception as exc:  # noqa: BLE001
            self.reason = f"model load failed: {exc}"

    def due(self, now: float) -> bool:
        return self.enabled and now >= self._next_allowed

    def infer(self, rgb: np.ndarray, now: float) -> Segmentation | None:
        if not self.enabled or self._session is None:
            return None
        if now < self._next_allowed:
            return None
        self._next_allowed = now + 1.0 / self.cfg.semantic_rate_hz
        started = time.monotonic()
        try:
            resized = nearest_resize(rgb, self.cfg.semantic_width, self.cfg.semantic_height)
            tensor = normalize_rgb(resized)
            outputs = self._session.run(None, {self._input_name: tensor})
            raw = np.asarray(outputs[0], dtype=np.float32)
            logits = np.squeeze(raw)
            if logits.ndim != 3:
                raise ValueError(f"unexpected segmentation output shape {raw.shape}")
            if logits.shape[0] != len(SEG_CLASSES):
                raise ValueError(
                    f"model has {logits.shape[0]} classes, expected {len(SEG_CLASSES)}"
                )
            probabilities = softmax(logits, axis=0)
            classes = np.argmax(probabilities, axis=0).astype(np.uint8)
            confidence = np.max(probabilities, axis=0)
            cost_at_model = np.zeros_like(classes, dtype=np.uint8)
            hazard_at_model = np.zeros_like(classes, dtype=bool)
            for index, spec in enumerate(SEG_CLASSES):
                mask = classes == index
                known = mask & (confidence >= self.cfg.semantic_conf_threshold)
                cost_at_model[known] = np.maximum(cost_at_model[known], spec.cost)
                if spec.hazard:
                    hazard_at_model[mask & (confidence >= self.cfg.hazard_conf_threshold)] = True
            cost = nearest_resize(
                cost_at_model[..., None], rgb.shape[1], rgb.shape[0]
            )[:, :, 0].astype(np.uint8)
            hazard = nearest_resize(
                hazard_at_model[..., None].astype(np.uint8) * 255,
                rgb.shape[1],
                rgb.shape[0],
            )[:, :, 0] > 127
            confidence_full = nearest_resize(
                confidence[..., None], rgb.shape[1], rgb.shape[0]
            )[:, :, 0]
            if self._ema is None or self._ema.shape != cost.shape:
                self._ema = cost.astype(np.float32)
            else:
                self._ema = (1.0 - self._ema_weight) * self._ema + self._ema_weight * cost
            smoothed = self._ema.astype(np.uint8)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            return Segmentation(smoothed, hazard, confidence_full, classes, elapsed_ms)
        except Exception:  # noqa: BLE001
            # One bad frame disables semantics for this process; geometry lives on.
            self.enabled = False
            self.reason = "inference failed; semantics disabled"
            return None


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def depth_selected_points(
    depth: np.ndarray,
    intr: Intrinsics,
    stride: int,
    min_range_m: float,
    max_range_m: float,
    class_cost: np.ndarray,
    hazard_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth pixels whose semantic cost is positive."""
    step = max(1, int(stride))
    z = depth[::step, ::step]
    u = np.arange(0, depth.shape[1], step, dtype=np.float32)[: z.shape[1]]
    v = np.arange(0, depth.shape[0], step, dtype=np.float32)[: z.shape[0]]
    uu, vv = np.meshgrid(u, v)
    valid = (
        np.isfinite(z)
        & (z >= min_range_m)
        & (z <= max_range_m)
        & (class_cost[: z.shape[0], : z.shape[1]] > 0)
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.uint8)
    zz = z[valid].astype(np.float64)
    x = (uu[valid].astype(np.float64) - intr.cx) * zz / intr.fx
    y = (vv[valid].astype(np.float64) - intr.cy) * zz / intr.fy
    points = np.stack((x, y, zz), axis=1)
    return points, class_cost[: z.shape[0], : z.shape[1]][valid].astype(np.uint8)