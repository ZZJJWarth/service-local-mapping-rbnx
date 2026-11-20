import numpy as np

from local_mapping_rbnx.semantics import SEG_CLASSES, softmax, depth_selected_points
from local_mapping_rbnx.sensors import Intrinsics


def test_class_contract():
    assert [spec.name for spec in SEG_CLASSES] == [
        "path",
        "grass",
        "vegetation",
        "water",
        "lava",
        "structure",
        "unknown",
    ]
    assert SEG_CLASSES[0].cost == 0
    assert all(spec.cost <= 100 for spec in SEG_CLASSES)
    assert sum(1 for spec in SEG_CLASSES if spec.hazard) == 4


def test_softmax_is_distribution():
    logits = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    probabilities = softmax(logits, axis=1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_depth_selected_points_only_positive_cost():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    cost = np.zeros((4, 4), dtype=np.uint8)
    cost[1, 1] = 50
    hazard = np.zeros((4, 4), dtype=bool)
    hazard[1, 1] = True
    intr = Intrinsics(4, 4, fx=1.0, fy=1.0, cx=2.0, cy=2.0)
    points, costs = depth_selected_points(
        depth, intr, 1, 0.2, 10.0, cost, hazard
    )
    assert points.shape == (1, 3)
    assert costs.tolist() == [50]
    assert np.allclose(points[0], [-2.0, -2.0, 2.0])


def test_depth_selected_points_empty_when_no_semantics():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    cost = np.zeros((4, 4), dtype=np.uint8)
    intr = Intrinsics(4, 4, fx=1.0, fy=1.0, cx=2.0, cy=2.0)
    points, costs = depth_selected_points(
        depth, intr, 1, 0.2, 10.0, cost, np.zeros((4, 4), dtype=bool)
    )
    assert points.shape == (0, 3)
    assert costs.shape == (0,)