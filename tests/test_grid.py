import numpy as np

from local_mapping_rbnx.grid import LETHAL, UNKNOWN, RollingGrid
from test_params import make_cfg


def window_indices(grid, x, y, robot=(0.0, 0.0)):
    grid.set_robot(*robot)
    col0 = int(round((grid.robot_x - grid.publish_size / 2.0 - grid.origin_x) / grid.res))
    row0 = int(round((grid.robot_y - grid.publish_size / 2.0 - grid.origin_y) / grid.res))
    col, row = grid.world_to_index(x, y)
    return int(row) - row0, int(col) - col0


def test_lidar_marks_endpoint_and_clears_ray():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    hits = grid.integrate_lidar(
        (0.0, 0.0), np.asarray([[3.0, 0.0, -1.0]]), 1.0, -1.62
    )
    assert hits == 1

    window = grid.window((0.0, 0.0))
    row, col = window_indices(grid, 3.0, 0.0)
    assert window.cost[row, col] == LETHAL

    row, col = window_indices(grid, 2.0, 0.0)
    assert window.cost[row, col] == 0
    assert window.occupancy[row, col] == 0


def test_lidar_ignores_points_outside_height_band():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    high = np.asarray([[3.0, 0.0, 3.0]])
    assert grid.integrate_lidar((0.0, 0.0), high, 1.0, -1.62) == 0
    col, row = grid.world_to_index(3.0, 0.0)
    assert grid.log_odds[int(row), int(col)] <= 0.0


def test_depth_separates_ground_from_obstacle():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    ground = np.asarray([[1.0, 1.0, -1.6]])
    assert grid.integrate_depth(ground, 1.0, -1.6) == 0
    col, row = grid.world_to_index(1.0, 1.0)
    assert np.isfinite(grid.ground_z[int(row), int(col)])

    step = np.asarray([[1.0, 1.0, -0.5]])
    assert grid.integrate_depth(step, 2.0, -1.6) == 1
    window = grid.window((0.0, 0.0))
    row, col = window_indices(grid, 1.0, 1.0)
    assert window.cost[row, col] == LETHAL


def test_depth_ignores_tall_structure_above_collision_band():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    canopy = np.asarray([[2.0, 2.0, 3.0]])
    assert grid.integrate_depth(canopy, 1.0, -1.6) == 0
    col, row = grid.world_to_index(2.0, 2.0)
    assert not np.isfinite(grid.obstacle_z[int(row), int(col)])


def test_semantic_never_lowers_lethal():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    grid.integrate_lidar((0.0, 0.0), np.asarray([[2.0, 0.0, -1.0]]), 1.0, -1.62)
    grid.integrate_semantic(np.asarray([[2.0, 0.0, -1.0]]), 15, 1.0)
    window = grid.window((0.0, 0.0))
    row, col = window_indices(grid, 2.0, 0.0)
    assert window.cost[row, col] == LETHAL


def test_semantic_adds_cost_to_free_cell():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    grid.integrate_lidar((0.0, 0.0), np.asarray([[3.0, 0.0, -1.0]]), 1.0, -1.62)
    grid.integrate_semantic(np.asarray([[2.0, 0.0, -1.0]]), 100, 1.0)
    window = grid.window((0.0, 0.0))
    row, col = window_indices(grid, 2.0, 0.0)
    assert window.cost[row, col] == 100


def test_window_outside_grid_is_unknown():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    window = grid.window((0.0, 0.0))
    assert window.cost.shape == (40, 40)
    assert np.all(window.cost == UNKNOWN)


def test_rolling_preserves_world_coordinates():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    grid.integrate_lidar((0.0, 0.0), np.asarray([[2.0, 0.0, -1.0]]), 1.0, -1.62)
    col, row = grid.world_to_index(2.0, 0.0)
    before = float(grid.log_odds[int(row), int(col)])
    assert before > 0.5

    window = grid.window((5.0, 0.0))
    col, row = grid.world_to_index(2.0, 0.0)
    assert float(grid.log_odds[int(row), int(col)]) == before
    assert abs(window.origin_x - 1.0) < 1e-9
    row, col = window_indices(grid, 2.0, 0.0, robot=(5.0, 0.0))
    assert window.cost[row, col] == LETHAL


def test_decay_clears_stale_obstacles():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    grid.integrate_lidar((0.0, 0.0), np.asarray([[2.0, 0.0, -1.0]]), 1.0, -1.62)
    col, row = grid.world_to_index(2.0, 0.0)
    grid.decay(2.0, 8.0 * 20.0)
    assert not np.isfinite(grid.obstacle_z[int(row), int(col)])
    window = grid.window((0.0, 0.0))
    row, col = window_indices(grid, 2.0, 0.0)
    assert window.cost[row, col] == 0