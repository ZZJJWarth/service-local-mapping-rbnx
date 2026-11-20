from types import SimpleNamespace

import numpy as np
import pytest

from local_mapping_rbnx import sensors


def camera_info(width=4, height=3, fx=2.0, fy=2.0, cx=1.5, cy=1.0):
    k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    return SimpleNamespace(width=width, height=height, k=k)


def test_intrinsics_from_camera_info():
    intr = sensors.Intrinsics.from_camera_info(camera_info())
    assert (intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy) == (
        4,
        3,
        2.0,
        2.0,
        1.5,
        1.0,
    )
    with pytest.raises(ValueError):
        sensors.Intrinsics.from_camera_info(SimpleNamespace(width=0, height=3, k=[]))
    with pytest.raises(ValueError):
        sensors.Intrinsics.from_camera_info(
            SimpleNamespace(width=4, height=3, k=[0, 0, 0, 0, 0, 0, 0, 0, 1])
        )


def test_backproject_centre_pixel():
    depth = np.full((3, 4), 2.0, dtype=np.float32)
    intr = sensors.Intrinsics(4, 3, fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    points = sensors.backproject_depth(depth, intr, 1, 0.2, 10.0)
    assert points.shape == (12, 3)
    assert np.allclose(points[:, 2], 2.0)
    centre = points[np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 1], 0.0)]
    assert centre.shape[0] == 1
    assert centre[0, 2] == pytest.approx(2.0)


def test_backproject_filters_range_and_nan():
    depth = np.asarray([[np.nan, 0.05, 5.0, 50.0]], dtype=np.float32)
    intr = sensors.Intrinsics(4, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    points = sensors.backproject_depth(depth, intr, 1, 0.2, 10.0)
    assert points.shape == (1, 3)
    assert points[0, 2] == pytest.approx(5.0)


def test_depth_decoding():
    raw = np.asarray([[1000, 2000, 3000]], dtype="<u2").tobytes()
    mm = sensors.depth_to_metres(raw, "16UC1", 1, 3, 6)
    assert np.allclose(mm, [[1.0, 2.0, 3.0]])

    metres = np.asarray([[1.5, 2.5, 3.5]], dtype="<f4").tobytes()
    m = sensors.depth_to_metres(metres, "32FC1", 1, 3, 12)
    assert np.allclose(m, [[1.5, 2.5, 3.5]])
    with pytest.raises(ValueError):
        sensors.depth_to_metres(metres, "rgb8", 1, 3, 12)


def test_scan_endpoints_filters_inf():
    endpoints = sensors.scan_endpoints(
        [1.0, float("inf"), 2.0], angle_min=0.0, angle_increment=np.pi / 2, range_min=0.1, range_max=10
    )
    assert endpoints.shape == (2, 3)
    assert np.allclose(endpoints[0], [1.0, 0.0, 0.0])
    assert np.allclose(endpoints[1], [-2.0, 0.0, 0.0], atol=1e-12)


def test_pointcloud_xyz_float32_with_padding():
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8],
            "itemsize": 16,
        }
    )
    raw = np.zeros(2, dtype=dtype)
    raw["x"] = [1.0, 2.0]
    raw["y"] = [3.0, 4.0]
    raw["z"] = [5.0, 6.0]
    fields = [
        SimpleNamespace(name=name, datatype=7, offset=offset)
        for name, offset in (("x", 0), ("y", 4), ("z", 8))
    ]
    points = sensors.pointcloud_xyz(
        raw.tobytes(), 16, 2, 1, 32, fields, False
    )
    assert np.allclose(points, [[1, 3, 5], [2, 4, 6]])

    with pytest.raises(ValueError):
        sensors.pointcloud_xyz(
            raw.tobytes(), 16, 2, 1, 32, [SimpleNamespace(name="x", datatype=7, offset=0)], False
        )


def test_decode_rgb_channel_order():
    rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8).tobytes()
    assert np.array_equal(sensors.decode_rgb(rgb, "rgb8", 1, 1, 3), [[[10, 20, 30]]])
    assert np.array_equal(sensors.decode_rgb(rgb, "bgr8", 1, 1, 3), [[[30, 20, 10]]])


def test_normalize_and_resize():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    resized = sensors.nearest_resize(image, 5, 5)
    assert resized.shape == (5, 5, 3)
    tensor = sensors.normalize_rgb(resized)
    assert tensor.shape == (1, 3, 5, 5)
    assert tensor.dtype == np.float32