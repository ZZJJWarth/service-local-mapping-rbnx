import math

import numpy as np

from local_mapping_rbnx import frames


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class Quat:
    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w


class Transform:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation


class Stamped:
    def __init__(self, translation, rotation):
        self.transform = Transform(translation, rotation)


def test_quaternion_matrix_yaw_90():
    rotation = frames.quaternion_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    point = frames.transform_points(
        np.asarray([[1.0, 0.0, 0.0]]), Vec(0.0, 0.0, 0.0), Quat(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    )
    assert np.allclose(point[0], [0.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(rotation @ np.asarray([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_transform_points_translation():
    point = frames.transform_points(
        np.asarray([[1.0, 2.0, 3.0]]), Vec(10.0, 20.0, 30.0), Quat(0.0, 0.0, 0.0, 1.0)
    )
    assert np.allclose(point[0], [11.0, 22.0, 33.0])


def test_compose_transforms():
    parent = Stamped(Vec(1.0, 0.0, 0.0), Quat(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
    child = Stamped(Vec(1.0, 0.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0))
    translation, rotation = frames.compose_transforms(parent, child)
    assert np.allclose(translation, [1.0, 1.0, 0.0], atol=1e-9)
    assert np.allclose(rotation, frames.quaternion_matrix(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)))


def test_matrix_round_trip():
    rotation = frames.quaternion_matrix(0.1, -0.2, 0.3, 0.9)
    x, y, z, w = frames.matrix_to_quaternion(rotation)
    assert np.allclose(frames.quaternion_matrix(x, y, z, w), rotation, atol=1e-9)


def test_invalid_quaternion():
    try:
        frames.quaternion_matrix(0.0, 0.0, 0.0, 0.0)
    except ValueError as exc:
        assert "quaternion" in str(exc)
    else:
        raise AssertionError("expected ValueError")