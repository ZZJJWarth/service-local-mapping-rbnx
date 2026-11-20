from pathlib import Path

import pytest

from local_mapping_rbnx import params

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(**overrides):
    config = {"sensor_providers": {"scan": "lidar0", "odom": "chassis0"}}
    config.update(overrides)
    return params.from_config(config, ROOT)


def test_defaults_are_valid():
    cfg = make_cfg()
    assert cfg.cells == 320
    assert cfg.publish_cells == 160
    assert cfg.map_frame == "minecraft_odom"
    assert cfg.base_frame == "minecraft_view_base"
    assert not cfg.semantic_enabled


def test_unknown_role_rejected():
    with pytest.raises(params.ConfigError, match="unknown sensor role"):
        params.from_config(
            {"sensor_providers": {"scan": "a", "odom": "b", "ground_truth": "c"}},
            ROOT,
        )


def test_unknown_key_rejected():
    with pytest.raises(params.ConfigError, match="unknown config key"):
        make_cfg(scan_topic="/minecraft/lidar/scan")


def test_odom_required():
    with pytest.raises(params.ConfigError, match="odom is required"):
        params.from_config({"sensor_providers": {"scan": "a"}}, ROOT)


def test_geometry_required():
    with pytest.raises(params.ConfigError, match="at least one"):
        params.from_config(
            {"sensor_providers": {"odom": "a", "rgb": "b"}}, ROOT
        )


def test_depth_requires_intrinsics():
    with pytest.raises(params.ConfigError, match="depth requires"):
        params.from_config(
            {"sensor_providers": {"odom": "a", "depth": "b"}}, ROOT
        )


def test_publish_larger_than_grid_rejected():
    with pytest.raises(params.ConfigError, match="publish_size_m"):
        make_cfg(grid_size_m=32, publish_size_m=48)


def test_frames_must_share_odom_frame():
    with pytest.raises(params.ConfigError, match="map_frame must equal"):
        make_cfg(map_frame="map", odom_frame="odom")


def test_semantic_requires_model_and_rgb():
    cfg = make_cfg(
        sensor_providers={
            "scan": "a",
            "odom": "b",
            "depth": "c",
            "intrinsics": "d",
            "rgb": "e",
        }
    )
    assert not cfg.semantic_enabled
    cfg = make_cfg(
        sensor_providers={
            "scan": "a",
            "odom": "b",
            "depth": "c",
            "intrinsics": "d",
            "rgb": "e",
        },
        semantic_model_path="models/test.onnx",
    )
    assert cfg.semantic_enabled
    assert cfg.semantic_model_path == str(ROOT / "models" / "test.onnx")


def test_integral_float_is_integer():
    cfg = make_cfg(depth_stride=4.0)
    assert cfg.depth_stride == 4
    with pytest.raises(params.ConfigError, match="exact integer"):
        make_cfg(depth_stride=4.5)