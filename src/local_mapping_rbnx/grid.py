# SPDX-License-Identifier: MulanPSL-2.0
"""Rolling 2.5D occupancy/elevation grid.

All coordinates are metres in the configured odometry frame. The grid keeps a
larger internal window than it publishes so short-range clearing and height
history survive while the robot moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from local_mapping_rbnx.params import MapConfig

LETHAL = 100
UNKNOWN = -1


@dataclass(frozen=True)
class Window:
    cost: np.ndarray
    occupancy: np.ndarray
    origin_x: float
    origin_y: float
    resolution: float
    obstacle_points: np.ndarray


class RollingGrid:
    def __init__(self, cfg: MapConfig) -> None:
        self.cfg = cfg
        self.n = cfg.cells
        self.res = cfg.resolution_m
        self.size = cfg.grid_size_m
        self.publish_n = cfg.publish_cells
        self.publish_size = cfg.publish_size_m
        self.origin_x = -self.size / 2.0
        self.origin_y = -self.size / 2.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_ground_z = float("nan")
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)
        self.ground_z = np.full((self.n, self.n), np.nan, dtype=np.float32)
        self.obstacle_z = np.full((self.n, self.n), np.nan, dtype=np.float32)
        self.seen = np.zeros((self.n, self.n), dtype=np.uint16)
        self.last_hit = np.zeros((self.n, self.n), dtype=np.float64)
        self.stale_obstacle = np.zeros((self.n, self.n), dtype=bool)
        self.semantic_cost = np.zeros((self.n, self.n), dtype=np.uint8)
        self.semantic_time = np.zeros((self.n, self.n), dtype=np.float64)

    # ── frame / index helpers ───────────────────────────────────────────────
    @property
    def center_x(self) -> float:
        return self.origin_x + self.size / 2.0

    @property
    def center_y(self) -> float:
        return self.origin_y + self.size / 2.0

    def world_to_index(self, x: np.ndarray | float, y: np.ndarray | float):
        cols = np.floor((np.asarray(x) - self.origin_x) / self.res).astype(np.int64)
        rows = np.floor((np.asarray(y) - self.origin_y) / self.res).astype(np.int64)
        return cols, rows

    def index_to_world(self, cols: np.ndarray, rows: np.ndarray):
        return (
            self.origin_x + (np.asarray(cols) + 0.5) * self.res,
            self.origin_y + (np.asarray(rows) + 0.5) * self.res,
        )

    def set_robot(self, x: float, y: float) -> None:
        self.robot_x = float(x)
        self.robot_y = float(y)
        self.ensure_centered()

    def ensure_centered(self) -> None:
        margin = max(self.res, (self.size - self.publish_size) / 4.0)
        if abs(self.robot_x - self.center_x) <= margin and abs(self.robot_y - self.center_y) <= margin:
            return
        shift_cols = int(round((self.robot_x - self.center_x) / self.res))
        shift_rows = int(round((self.robot_y - self.center_y) / self.res))
        if shift_cols:
            self._shift_axis(cols=shift_cols)
        if shift_rows:
            self._shift_axis(rows=shift_rows)
        self.origin_x += shift_cols * self.res
        self.origin_y += shift_rows * self.res

    def reset(self, robot_xy: tuple[float, float] | None = None) -> None:
        """Clear all grid layers and recenter around the robot."""
        if robot_xy is not None:
            self.robot_x = float(robot_xy[0])
            self.robot_y = float(robot_xy[1])
        self.origin_x = self.robot_x - self.size / 2.0
        self.origin_y = self.robot_y - self.size / 2.0
        self.robot_ground_z = float("nan")
        self.log_odds.fill(0.0)
        self.ground_z.fill(np.nan)
        self.obstacle_z.fill(np.nan)
        self.seen.fill(0)
        self.last_hit.fill(0.0)
        self.stale_obstacle.fill(False)
        self.semantic_cost.fill(0)
        self.semantic_time.fill(0.0)

    def _shift_axis(self, cols: int = 0, rows: int = 0) -> None:
        for name, fill in (
            ("log_odds", 0.0),
            ("ground_z", np.nan),
            ("obstacle_z", np.nan),
            ("seen", 0),
            ("last_hit", 0.0),
            ("stale_obstacle", False),
            ("semantic_cost", 0),
            ("semantic_time", 0.0),
        ):
            array = getattr(self, name)
            if abs(rows) >= self.n or abs(cols) >= self.n:
                array.fill(fill)
                continue
            shifted = np.roll(array, (-rows, -cols), axis=(0, 1))
            if rows > 0:
                shifted[-rows:, :] = fill
            elif rows < 0:
                shifted[:-rows, :] = fill
            if cols > 0:
                shifted[:, -cols:] = fill
            elif cols < 0:
                shifted[:, :-cols] = fill
            setattr(self, name, shifted)

    # ── integration ─────────────────────────────────────────────────────────
    def integrate_lidar(
        self,
        origin_xy: tuple[float, float],
        endpoints: np.ndarray,
        stamp: float,
        ground_reference_z: float,
    ) -> int:
        """Integrate transformed lidar endpoints with height-aware, once-per-frame ray clearing."""
        if endpoints.size == 0:
            return 0
        points = np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        if points.size == 0:
            return 0
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        self.robot_ground_z = float(ground_reference_z)
        accepted = 0
        min_z = ground_reference_z + self.cfg.obstacle_min_height_m
        max_z = ground_reference_z + self.cfg.obstacle_max_height_m

        hit_flat: set[int] = set()
        clear_min_z: dict[int, float] = {}

        for x, y, z in points:
            dist = math.hypot(x - ox, y - oy)
            if dist < self.cfg.lidar_min_range_m or dist > self.cfg.lidar_max_range_m:
                continue
            self._collect_ray_cells(ox, oy, x, y, float(z), clear_min_z)
            if z < min_z or z > max_z:
                continue
            cols, rows = self.world_to_index(x, y)
            if not (0 <= cols < self.n and 0 <= rows < self.n):
                continue
            c_int, r_int = int(cols), int(rows)
            hit_flat.add(r_int * self.n + c_int)
            self._mark_occupied(c_int, r_int, float(z), stamp)
            accepted += 1

        if clear_min_z:
            self._apply_ray_clearing(clear_min_z, hit_flat)
        return accepted

    def _collect_ray_cells(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        ray_z: float,
        clear_min_z: dict[int, float],
    ) -> None:
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist <= self.res:
            return
        steps = int(math.ceil(dist / self.res))
        if steps < 2:
            return
        fractions = np.linspace(0.0, 1.0, steps + 1, endpoint=False)[1:-1]
        if fractions.size == 0:
            return
        xs = x0 + dx * fractions
        ys = y0 + dy * fractions
        cols, rows = self.world_to_index(xs, ys)
        valid = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        if not np.any(valid):
            return
        flat = np.unique(rows[valid] * self.n + cols[valid])
        for idx in flat.tolist():
            prev = clear_min_z.get(idx)
            if prev is None or ray_z < prev:
                clear_min_z[idx] = ray_z

    def _apply_ray_clearing(
        self, clear_min_z: dict[int, float], hit_flat: set[int]
    ) -> None:
        for idx in hit_flat:
            clear_min_z.pop(idx, None)
        if not clear_min_z:
            return
        flat = np.fromiter(clear_min_z.keys(), dtype=np.int64, count=len(clear_min_z))
        ray_zs = np.fromiter(clear_min_z.values(), dtype=np.float32, count=len(clear_min_z))
        r = flat // self.n
        c = flat % self.n

        # A high-ring ray passing over a shorter obstacle must not clear it.
        obs_z = self.obstacle_z[r, c]
        allow = ~np.isfinite(obs_z) | (ray_zs <= obs_z + 0.15)
        if not np.any(allow):
            return
        r = r[allow]
        c = c[allow]
        self.seen[r, c] = np.minimum(
            self.seen[r, c].astype(np.uint32) + 1, 65535
        ).astype(np.uint16)
        self.log_odds[r, c] = np.maximum(
            self.log_odds[r, c] - self.cfg.log_odds_free, self.cfg.log_odds_min
        )
        cleared = self.log_odds[r, c] <= -1.5
        if np.any(cleared):
            rc_r, rc_c = r[cleared], c[cleared]
            self.stale_obstacle[rc_r, rc_c] = False
            self.obstacle_z[rc_r, rc_c] = np.nan

    def _mark_occupied(self, col: int, row: int, z: float, stamp: float) -> None:
        self.seen[row, col] = min(int(self.seen[row, col]) + 1, 65535)
        self.log_odds[row, col] = min(
            float(self.log_odds[row, col]) + self.cfg.log_odds_occupied,
            self.cfg.log_odds_max,
        )
        if self.log_odds[row, col] >= 0.5:
            self.stale_obstacle[row, col] = True
        current = self.obstacle_z[row, col]
        self.obstacle_z[row, col] = z if not np.isfinite(current) else max(float(current), z)
        self.last_hit[row, col] = stamp

    def integrate_depth(
        self,
        points: np.ndarray,
        stamp: float,
        robot_ground_z: float,
    ) -> int:
        """Cluster map-frame depth returns per cell into ground and obstacles."""
        if points.size == 0:
            return 0
        self.robot_ground_z = float(robot_ground_z)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.size == 0:
            return 0
        cols, rows = self.world_to_index(pts[:, 0], pts[:, 1])
        valid = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        if not np.any(valid):
            return 0
        pts = pts[valid]
        cols = cols[valid]
        rows = rows[valid]
        flat = rows * self.n + cols
        order = np.argsort(flat, kind="stable")
        flat = flat[order]
        zs = pts[order, 2]
        unique, starts = np.unique(flat, return_index=True)
        ends = np.r_[starts[1:], len(flat)]

        # Seed robot's immediate ground cell so 2.5D elevation smoothly propagates up/down stairs.
        rc_col, rc_row = self.world_to_index(self.robot_x, self.robot_y)
        if 0 <= int(rc_col) < self.n and 0 <= int(rc_row) < self.n:
            cur_rg = self.ground_z[int(rc_row), int(rc_col)]
            self.ground_z[int(rc_row), int(rc_col)] = (
                float(robot_ground_z)
                if not np.isfinite(cur_rg)
                else 0.65 * float(cur_rg) + 0.35 * float(robot_ground_z)
            )

        # Process cells from nearest to farthest so neighbor ground_z climbs stairs step-by-step.
        u_rows = unique // self.n
        u_cols = unique % self.n
        u_wx, u_wy = self.index_to_world(u_cols, u_rows)
        dist2 = (u_wx - self.robot_x) ** 2 + (u_wy - self.robot_y) ** 2
        dist_order = np.argsort(dist2, kind="stable")

        accepted = 0
        percentile = float(self.cfg.ground_percentile)
        step_h = float(self.cfg.step_height_m)
        for k in dist_order.tolist():
            r = int(u_rows[k])
            c = int(u_cols[k])
            cell_z = zs[starts[k]:ends[k]]
            z_min = float(np.min(cell_z))
            z_max = float(np.max(cell_z))
            span = z_max - z_min

            tracked = self.ground_z[r, c]
            if np.isfinite(tracked):
                local_ground = float(tracked)
            else:
                # 5x5 (+-0.4m) neighborhood bridges depth_stride gaps on ascending/descending stairs;
                # pick the neighbor elevation closest to this surface's base z_min.
                r0, r1 = max(0, r - 2), min(self.n, r + 3)
                c0, c1 = max(0, c - 2), min(self.n, c + 3)
                nb = self.ground_z[r0:r1, c0:c1]
                nb_fin = nb[np.isfinite(nb)]
                if nb_fin.size > 0:
                    local_ground = float(nb_fin[int(np.argmin(np.abs(nb_fin - z_min)))])
                else:
                    local_ground = float(robot_ground_z)

            # A vertical face (span > step_h) or an elevated block whose base rises > step_h
            # above neighbor ground is an obstacle; a flat/stepped tread whose base connects
            # within step_h (OR rises along a <= 49 deg stair ramp through the camera's near-field
            # blind spot where LiDAR has already confirmed free passage, log_odds < 0) updates ground_z!
            lidar_cleared_ramp = (
                span <= step_h
                and float(self.log_odds[r, c]) < -0.05
                and (z_min - float(robot_ground_z)) > 0.0
                and (z_min - float(robot_ground_z)) <= math.hypot(float(u_wx[k]) - self.robot_x, float(u_wy[k]) - self.robot_y) * 1.15 + 0.25
            )
            at_robot_floor = (
                span <= step_h
                and abs(z_min - float(robot_ground_z)) <= step_h
                and abs(z_max - float(robot_ground_z)) <= step_h
            )
            if not lidar_cleared_ramp and not at_robot_floor and ((z_min - local_ground) > step_h or span > step_h):
                if (z_max - local_ground) > step_h:
                    # Upward vertical wall / obstacle rising above local_ground.
                    if (z_max - local_ground) <= self.cfg.obstacle_max_height_m:
                        self._mark_occupied(c, r, z_max, stamp)
                        accepted += 1
                    continue
                # Downward vertical pit wall: only trust if a robust cluster (>= 4 pts, median
                # clearly below local_ground) confirms a real drop — never a single noisy outlier!
                if cell_z.size >= 4 and float(np.percentile(cell_z, 50.0)) < local_ground - step_h:
                    pit_z = float(np.percentile(cell_z, 25.0))
                    self.ground_z[r, c] = pit_z if not np.isfinite(tracked) else (
                        0.4 * float(tracked) + 0.6 * pit_z
                    )
                    self.seen[r, c] = min(int(self.seen[r, c]) + 1, 65535)
                continue
            eff_pct = 80.0 if span > 0.15 else percentile
            candidate = float(np.percentile(cell_z, eff_pct))
            if not np.isfinite(tracked):
                new_g = candidate
            elif at_robot_floor and abs(float(tracked) - candidate) > step_h:
                new_g = candidate
            elif (float(tracked) - candidate) > step_h and cell_z.size >= 3:
                new_g = 0.35 * float(tracked) + 0.65 * candidate
            else:
                new_g = 0.75 * float(tracked) + 0.25 * candidate
            self.ground_z[r, c] = new_g
            self.seen[r, c] = min(int(self.seen[r, c]) + 1, 65535)
            # Clear stale obstacle ghosts (e.g. old stair-riser hits) once depth confirms
            # this cell is a walkable tread/floor at or near the old obstacle height.
            old_obs = self.obstacle_z[r, c]
            if np.isfinite(old_obs) and old_obs <= new_g + step_h:
                self.log_odds[r, c] = max(
                    float(self.log_odds[r, c]) - self.cfg.log_odds_occupied,
                    self.cfg.log_odds_min,
                )
                if self.log_odds[r, c] < 0.2:
                    self.stale_obstacle[r, c] = False
                    self.obstacle_z[r, c] = np.nan
        return accepted

    def integrate_semantic(self, points: np.ndarray, cost: int, stamp: float) -> int:
        if points.size == 0 or cost <= 0:
            return 0
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cols, rows = self.world_to_index(pts[:, 0], pts[:, 1])
        valid = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        if not np.any(valid):
            return 0
        cols, rows = cols[valid], rows[valid]
        flat = np.unique(rows * self.n + cols)
        r = flat // self.n
        c = flat % self.n
        self.semantic_cost[r, c] = np.maximum(self.semantic_cost[r, c], np.uint8(cost))
        self.semantic_time[r, c] = stamp
        return int(flat.size)

    def decay(self, now: float, dt: float) -> None:
        if self.cfg.decay_seconds <= 0 or dt <= 0:
            return
        factor = math.exp(-dt / self.cfg.decay_seconds)
        positive = self.log_odds > 0
        self.log_odds[positive] = np.maximum(
            self.log_odds[positive] * factor, 0.0
        )
        stale = self.log_odds < 0.05
        self.obstacle_z[stale] = np.nan
        self.semantic_cost = (self.semantic_cost.astype(np.float32) * factor).astype(np.uint8)
        self.semantic_cost[self.semantic_cost < 5] = 0

    # ── publication ─────────────────────────────────────────────────────────
    def window(self, robot_xy: tuple[float, float]) -> Window:
        self.set_robot(*robot_xy)
        col0 = int(round((self.robot_x - self.publish_size / 2.0 - self.origin_x) / self.res))
        row0 = int(round((self.robot_y - self.publish_size / 2.0 - self.origin_y) / self.res))
        origin_x = self.origin_x + col0 * self.res
        origin_y = self.origin_y + row0 * self.res

        cost = np.full((self.publish_n, self.publish_n), UNKNOWN, dtype=np.int8)
        occupancy = np.full((self.publish_n, self.publish_n), UNKNOWN, dtype=np.int8)
        obstacle_points: list[tuple[float, float, float]] = []

        src_c0 = max(0, col0)
        src_r0 = max(0, row0)
        src_c1 = min(self.n, col0 + self.publish_n)
        src_r1 = min(self.n, row0 + self.publish_n)
        if src_c1 <= src_c0 or src_r1 <= src_r0:
            return Window(cost, occupancy, origin_x, origin_y, self.res, np.zeros((0, 3)))

        dst_c0 = src_c0 - col0
        dst_r0 = src_r0 - row0
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        sl_src = np.s_[src_r0:src_r1, src_c0:src_c1]
        sl_dst = np.s_[dst_r0:dst_r1, dst_c0:dst_c1]

        log_odds = self.log_odds[sl_src]
        seen = self.seen[sl_src]
        obs_z_win = self.obstacle_z[sl_src]
        # Height-band filter: ignore stale obstacles from a different floor/elevation slice
        # when the robot climbs or descends stairs.
        if math.isfinite(self.robot_ground_z):
            z_lo = self.robot_ground_z + self.cfg.obstacle_min_height_m - 0.35
            z_hi = self.robot_ground_z + self.cfg.obstacle_max_height_m + 0.35
            in_height_band = ~np.isfinite(obs_z_win) | ((obs_z_win >= z_lo) & (obs_z_win <= z_hi))
        else:
            in_height_band = np.ones_like(log_odds, dtype=bool)
        occupied = (seen > 0) & (log_odds >= 0.5) & in_height_band
        free = (seen > 0) & ((log_odds <= 0.1) | ~in_height_band)
        uncertain = (seen > 0) & ~occupied & ~free
        block = cost[sl_dst]
        block[...] = UNKNOWN
        block[free] = 0
        block[uncertain] = 45

        if self.cfg.stale_obstacle_cost > 0:
            cols_idx = np.arange(src_c0, src_c1, dtype=np.float32)
            rows_idx = np.arange(src_r0, src_r1, dtype=np.float32)
            wx_axis = self.origin_x + (cols_idx + 0.5) * self.res
            wy_axis = self.origin_y + (rows_idx + 0.5) * self.res
            dist2 = (wx_axis[None, :] - self.robot_x) ** 2 + (
                wy_axis[:, None] - self.robot_y
            ) ** 2
            in_obs = dist2 <= (self.cfg.observation_radius_m ** 2)
            fresh_occupied = occupied & in_obs
            stale_mask = ((occupied & ~in_obs) | (
                self.stale_obstacle[sl_src] & in_height_band & ~fresh_occupied
            ))
            block[stale_mask] = np.int8(self.cfg.stale_obstacle_cost)
            block[fresh_occupied] = LETHAL
        else:
            fresh_occupied = occupied
            block[fresh_occupied] = LETHAL

        semantic = self.semantic_cost[sl_src].astype(np.int16)
        semantic[fresh_occupied] = 0
        block_sem = cost[sl_dst]
        semantic_mask = semantic > 0
        block_sem[semantic_mask] = np.maximum(
            block_sem[semantic_mask], semantic[semantic_mask].astype(np.int8)
        )

        # 2.5D Traversability (depth-camera layer; near-field >= 1.55 m pits are owned by the
        # exact downward-pitched LiDAR cliff scanner with zero lip-occlusion artifacts):
        #  - <= 1.5 m step (0.5 m stairs / 1-block steps): fully walkable (cost 0)
        #  - 1.5..drop_off_height_m (e.g. occluded downward staircase lip seen from horizontal camera):
        #    gentle caution (cost 25) so robot walks up to the edge and lets the downward LiDAR
        #    verify whether it is walkable stairs (occ=0) or a >= 2-block pit (occ=LETHAL)
        #  - > drop_off_height_m (default 3.5 m, unmistakable rooftop/cliff drop): LETHAL (100)
        ground = self.ground_z[sl_src]
        max_step = self._max_neighbor_step(ground)
        drop_h = float(self.cfg.drop_off_height_m)
        cliff = np.isfinite(max_step) & (max_step > drop_h)
        block_sem[cliff] = LETHAL
        caution_drop = np.isfinite(max_step) & (max_step > 1.50) & (max_step <= drop_h)
        block_sem[caution_drop] = np.maximum(block_sem[caution_drop], 25)
        # Macro slope over ~0.8 m span: gentle preference for flat ground without blocking stairs.
        macro_tan = self._macro_slope_tan(ground)
        steep_step_guard = max(1.50, float(self.cfg.step_height_m) + 0.95)
        steep = (
            np.isfinite(macro_tan)
            & (macro_tan > math.tan(math.radians(self.cfg.slope_max_deg)))
            & np.isfinite(max_step)
            & (max_step > steep_step_guard)
            & ~cliff
        )
        block_sem[steep] = np.maximum(block_sem[steep], 45)

        if self.cfg.wall_aversion_radius_m > 0.0 and self.cfg.wall_aversion_max_cost > 0:
            repulsion = self._wall_aversion(block >= LETHAL)
            apply_mask = (block >= 0) & (block < LETHAL) & (repulsion > 0)
            block[apply_mask] = np.maximum(block[apply_mask], repulsion[apply_mask])

        prob = 1.0 / (1.0 + np.exp(-log_odds.astype(np.float32)))
        occ = occupancy[sl_dst]
        occ[...] = UNKNOWN
        known = seen > 0
        occ[known] = 0
        occ[known & (prob >= 0.65)] = 100
        occ[known & (prob >= 0.85)] = LETHAL

        hit_r, hit_c = np.where(fresh_occupied)
        if hit_r.size:
            wx, wy = self.index_to_world(hit_c + src_c0, hit_r + src_r0)
            wz = self.obstacle_z[sl_src][fresh_occupied]
            obstacle_points = list(zip(wx.tolist(), wy.tolist(), wz.tolist()))
        obstacle_array = (
            np.asarray(obstacle_points, dtype=np.float64)
            if obstacle_points else np.zeros((0, 3), dtype=np.float64)
        )
        return Window(cost, occupancy, origin_x, origin_y, self.res, obstacle_array)

    def _wall_aversion(self, lethal_mask: np.ndarray) -> np.ndarray:
        """Compute soft repulsion cost around lethal cells via separable 2-pass EDT."""
        out = np.zeros(lethal_mask.shape, dtype=np.int8)
        if not np.any(lethal_mask):
            return out
        radius_m = float(self.cfg.wall_aversion_radius_m)
        max_cost = int(self.cfg.wall_aversion_max_cost)
        rad_cells = int(math.ceil(radius_m / self.res))
        if rad_cells <= 0 or max_cost <= 0:
            return out
        h, w = lethal_mask.shape
        inf = float((rad_cells + 1) ** 2 + 1)

        # Pass 1: horizontal minimum squared cell distance to any lethal cell.
        dx2 = np.where(lethal_mask, 0.0, inf).astype(np.float32)
        for dc in range(1, rad_cells + 1):
            c2 = float(dc * dc)
            if dc < w:
                left = np.where(lethal_mask[:, :-dc], c2, inf)
                dx2[:, dc:] = np.minimum(dx2[:, dc:], left)
                right = np.where(lethal_mask[:, dc:], c2, inf)
                dx2[:, :-dc] = np.minimum(dx2[:, :-dc], right)

        # Pass 2: vertical minimum squared cell distance (exact 2D squared EDT).
        d2 = dx2.copy()
        for dr in range(1, rad_cells + 1):
            r2 = float(dr * dr)
            if dr < h:
                d2[dr:, :] = np.minimum(d2[dr:, :], dx2[:-dr, :] + r2)
                d2[:-dr, :] = np.minimum(d2[:-dr, :], dx2[dr:, :] + r2)

        dist_m = np.sqrt(d2) * float(self.res)
        band = (dist_m > 0.0) & (dist_m <= radius_m)
        if np.any(band):
            falloff = np.power(1.0 - dist_m[band] / radius_m, 1.5)
            out[band] = np.clip(np.rint(max_cost * falloff), 1, max_cost).astype(np.int8)
        return out

    def _max_neighbor_step(self, ground: np.ndarray) -> np.ndarray:
        """Maximum absolute elevation step |Δz| between each cell and its 4-neighbors."""
        step = np.full(ground.shape, np.nan, dtype=np.float32)
        if ground.shape[0] > 1:
            dv = np.abs(np.diff(ground, axis=0))
            step[:-1, :] = np.fmax(step[:-1, :], dv)
            step[1:, :] = np.fmax(step[1:, :], dv)
        if ground.shape[1] > 1:
            dh = np.abs(np.diff(ground, axis=1))
            step[:, :-1] = np.fmax(step[:, :-1], dh)
            step[:, 1:] = np.fmax(step[:, 1:], dh)
        return step

    def _macro_slope_tan(self, ground: np.ndarray) -> np.ndarray:
        """Macroscopic terrain slope tan(theta) across ~0.8 m span (smooth over stair risers)."""
        k = max(1, int(round(0.8 / max(0.05, self.res))))
        span_m = float(k) * float(self.res)
        slope = np.full(ground.shape, np.nan, dtype=np.float32)
        if ground.shape[0] > k:
            dv = np.abs(ground[k:, :] - ground[:-k, :]) / span_m
            slope[:-k, :] = np.fmax(slope[:-k, :], dv)
            slope[k:, :] = np.fmax(slope[k:, :], dv)
        if ground.shape[1] > k:
            dh = np.abs(ground[:, k:] - ground[:, :-k]) / span_m
            slope[:, :-k] = np.fmax(slope[:, :-k], dh)
            slope[:, k:] = np.fmax(slope[:, k:], dh)
        return slope

    def _slope(self, ground: np.ndarray) -> np.ndarray:
        return self._macro_slope_tan(ground)

    def stats(self) -> dict[str, int]:
        return {
            "occupied_cells": int(np.count_nonzero(self.log_odds >= 0.5)),
            "free_cells": int(np.count_nonzero((self.seen > 0) & (self.log_odds <= 0.1))),
            "stale_obstacle_cells": int(np.count_nonzero(self.stale_obstacle)),
            "semantic_cells": int(np.count_nonzero(self.semantic_cost > 0)),
        }