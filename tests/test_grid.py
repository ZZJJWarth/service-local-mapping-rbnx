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


def test_wall_aversion_adds_decaying_repulsion_cost_around_walls():
    grid = RollingGrid(
        make_cfg(
            grid_size_m=16,
            publish_size_m=8,
            wall_aversion_radius_m=1.4,
            wall_aversion_max_cost=65,
        )
    )
    grid.integrate_lidar((0.0, 0.0), np.asarray([[3.0, 0.0, -1.0]]), 1.0, -1.62)
    window = grid.window((0.0, 0.0))

    r_wall, c_wall = window_indices(grid, 3.0, 0.0)
    r_near, c_near = window_indices(grid, 2.6, 0.0)   # 0.4 m from wall
    r_mid, c_mid = window_indices(grid, 2.0, 0.0)     # 1.0 m from wall
    r_far, c_far = window_indices(grid, 1.0, 0.0)     # 2.0 m from wall

    assert window.cost[r_wall, c_wall] == LETHAL
    assert 0 < window.cost[r_mid, c_mid] < window.cost[r_near, c_near] < LETHAL
    assert window.cost[r_far, c_far] == 0


def test_stale_obstacle_persists_outside_observation_radius_and_clears_on_reobservation():
    grid = RollingGrid(
        make_cfg(
            grid_size_m=48,
            publish_size_m=32,
            observation_radius_m=5.0,
            stale_obstacle_cost=82,
        )
    )
    # Observe a wall at (3.0, 0.0) while robot is at (0.0, 0.0).
    grid.integrate_lidar((0.0, 0.0), np.asarray([[3.0, 0.0, -1.0]]), 1.0, -1.62)
    win_near = grid.window((0.0, 0.0))
    r, c = window_indices(grid, 3.0, 0.0, robot=(0.0, 0.0))
    assert win_near.cost[r, c] == LETHAL

    # Robot moves away so (3.0, 0.0) is outside the 5 m observation radius
    # (distance 10 m) and decays; it must remain stale (82), not free or unknown.
    grid.decay(2.0, 8.0 * 20.0)
    win_far = grid.window((-7.0, 0.0))
    r, c = window_indices(grid, 3.0, 0.0, robot=(-7.0, 0.0))
    assert win_far.cost[r, c] == 82

    # Robot returns and actively observes sustained rays passing through (3.0, 0.0)
    # to a farther wall at (6.0, 0.0) -> (3.0, 0.0) is cleared back to free (0).
    for i in range(5):
        grid.integrate_lidar((0.0, 0.0), np.asarray([[6.0, 0.0, -1.0]]), 3.0 + i * 0.1, -1.62)
    win_clear = grid.window((0.0, 0.0))
    r, c = window_indices(grid, 3.0, 0.0, robot=(0.0, 0.0))
    assert win_clear.cost[r, c] == 0


def test_high_ring_ray_does_not_clear_low_obstacle():
    grid = RollingGrid(
        make_cfg(
            grid_size_m=16,
            publish_size_m=8,
            log_odds_occupied=1.2,
            log_odds_free=0.15,
            stale_obstacle_cost=82,
        )
    )
    # Low ring (-1.0 m) hits a 1-block wall at (3.0, 0.0), while high ring (-0.1 m)
    # shoots over it to a building at (7.0, 0.0).
    for i in range(6):
        pts = np.asarray([[3.0, 0.0, -1.0], [7.0, 0.0, -0.1]])
        grid.integrate_lidar((0.0, 0.0), pts, 1.0 + i * 0.1, -1.62)
    # Even if subsequent frames only see the high ring shooting over the low wall,
    # the low wall must NOT be erased by the higher ray.
    for i in range(15):
        pts = np.asarray([[7.0, 0.0, -0.1]])
        grid.integrate_lidar((0.0, 0.0), pts, 2.0 + i * 0.1, -1.62)
    win = grid.window((0.0, 0.0))
    r, c = window_indices(grid, 3.0, 0.0)
    assert win.cost[r, c] == LETHAL


def test_brief_stray_rays_during_turn_do_not_erase_confirmed_obstacle():
    grid = RollingGrid(
        make_cfg(
            grid_size_m=16,
            publish_size_m=8,
            log_odds_occupied=1.2,
            log_odds_free=0.15,
            stale_obstacle_cost=82,
        )
    )
    # Confirm obstacle over 4 frames (log_odds -> 4.8).
    for i in range(4):
        grid.integrate_lidar((0.0, 0.0), np.asarray([[3.0, 0.0, -1.0]]), 1.0 + i * 0.1, -1.62)
    # 3 frames of stray low rays passing through during a turn must not erase it.
    for i in range(3):
        grid.integrate_lidar((0.0, 0.0), np.asarray([[6.0, 0.0, -1.0]]), 2.0 + i * 0.1, -1.62)
    win = grid.window((0.0, 0.0))
    r, c = window_indices(grid, 3.0, 0.0)
    assert win.cost[r, c] == LETHAL


def test_rolling_clears_newly_exposed_edge_without_wraparound():
    grid = RollingGrid(make_cfg(grid_size_m=16, publish_size_m=8))
    # Place an obstacle near the -X edge (-7.0, 0.0) of the [-8, +8] internal grid.
    grid.integrate_lidar((0.0, 0.0), np.asarray([[-7.0, 0.0, -1.0]]), 1.0, -1.62)
    # Move robot +4 m in +X so the internal grid shifts +4 m (old -7.0 scrolls out).
    grid.set_robot(4.0, 0.0)
    # The newly exposed +X edge (e.g. +9.0..+11.8) must be empty, NOT wrapped-around data.
    col_new, row_new = grid.world_to_index(9.0, 0.0)
    assert grid.seen[int(row_new), int(col_new)] == 0
    assert grid.log_odds[int(row_new), int(col_new)] == 0.0


def test_reset_clears_all_grid_layers_and_recenters():
    grid = RollingGrid(
        make_cfg(grid_size_m=16, publish_size_m=8, stale_obstacle_cost=82)
    )
    grid.integrate_lidar((0.0, 0.0), np.asarray([[3.0, 0.0, -1.0]]), 1.0, -1.62)
    assert grid.stats()["occupied_cells"] > 0
    assert grid.stats()["stale_obstacle_cells"] > 0

    grid.reset((10.0, -5.0))
    assert grid.stats()["occupied_cells"] == 0
    assert grid.stats()["free_cells"] == 0
    assert grid.stats()["stale_obstacle_cells"] == 0
    assert abs(grid.center_x - 10.0) < 1e-6
    assert abs(grid.center_y - (-5.0)) < 1e-6