"""
What the cars sense: the racing-only lidar update (walls-only copy, wall /
car crash flags, ray-cast cull), the pairwise geometry between cars (once
per step), and the observation vector built for all cars at once.
"""

import math

import numpy as np
from numba import njit

from f1tenth_gym.envs.lidar import check_ttc_jit, ray_cast

from racer.track import SBETA, SDELTA, SV

# Far outside the map: where cars out of the race are parked (they also
# vanish from the render window).
CLEARED_POSE = (1.0e5, 1.0e5)


# -------------------------------------------------------------------------
# Lidar
# -------------------------------------------------------------------------

@njit(cache=True)
def _cast_cars(origin, scan, angles, verts, centres, me, radius, half_fov, max_range):
    """Ray-casts the other cars into one car's scan - only those a beam could
    hit: nearer than the walls-only scan's longest beam (beams only get
    shorter) and not entirely outside the field of view. The gym looped over
    every beam x 4 edges for every car, also the ones behind (a beam only
    hits forwards): 11 ms of a 16 ms step on a 22-car grid. Same scans."""
    for k in range(len(centres)):
        if k == me:
            continue
        dx = centres[k, 0] - origin[0]
        dy = centres[k, 1] - origin[1]
        dist = math.hypot(dx, dy)
        if dist >= max_range + radius + 1.0:
            continue
        bearing = abs((math.atan2(dy, dx) - origin[2] + math.pi) % (2 * math.pi) - math.pi)
        if dist > radius + 0.5 and bearing - math.asin(min(1.0, radius / max(dist, 1e-9))) > half_fov:
            continue
        scan = ray_cast(origin, scan, angles, verts[k])
    return scan


def car_vertices(cfg, xy, yaw):
    """Body corners (cars, 4, 2) in the gym's order (rear-left, rear-right,
    front-right, front-left), from rear-axle poses."""
    v = cfg.vehicle
    c, s = np.cos(yaw), np.sin(yaw)
    centre = xy + v.collision_body_center_x * np.stack([c, s], axis=1)
    hl, hw = v.length / 2, v.width / 2
    local = np.array([[-hl, hw], [-hl, -hw], [hl, -hw], [hl, hw]])
    rot = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)   # (cars, 2, 2)
    return centre[:, None, :] + np.einsum("nij,kj->nki", rot, local), centre


def racing_scan_update(env):
    """Replacement for F110Simulator._update_scans: only cars still racing
    are scanned and ray-cast (cars out of the race are off the track). Per
    racing car: the walls-only scan -> the gym's TTC check (a hit is a wall
    crash: speed zeroed, collision set, env.wall_hit); a copy of that scan
    kept in env.wall_scans (ray_cast edits its array in place); the other
    cars ray-cast in (culled); noise from the car's own rng; float32."""
    cfg = env.cfg

    def update_scans(sim):
        racing = np.flatnonzero(~env.done)
        if len(racing) == 0:
            return
        st = sim.state
        params = sim.vehicle_params
        radius = 0.5 * float(np.hypot(params.length, params.width))
        cache0 = sim.scan_cache[0]
        half_fov = float(np.max(np.abs(cache0.angles))) + np.deg2rad(2.0)
        poses = st.poses[racing].astype(np.float64)
        verts, centres = car_vertices(cfg, poses[:, :2], poses[:, 2])
        tx, ty, tyaw = sim.config.lidar_config.base_link_to_lidar_tf
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        origins = np.stack([poses[:, 0] + tx * c - ty * s, poses[:, 1] + tx * s + ty * c, poses[:, 2] + tyaw], axis=1)
        for k, i in enumerate(racing):
            scanner = sim.scan_sims[i]
            cache = sim.scan_cache[i]
            clean = scanner.scan(origins[k], rng=None)
            hit = check_ttc_jit(clean, st.standard_state[i, SV], cache.angles, cache.cosines,
                                cache.side_distances, sim.ttc_threshold)
            env.wall_hit[i] = hit
            if hit:
                st.state[i, 3:] = 0.0
                st.collisions[i] = 1.0
            else:
                st.collisions[i] = 0.0
            env.wall_scans[i] = clean
            adjusted = _cast_cars(origins[k], clean.copy(), cache.angles, verts, centres, k, radius, half_fov,
                                  float(clean.max()))
            sim._adjusted_scans[i] = adjusted
            noisy = adjusted + sim.scan_rngs[i].normal(0.0, scanner.std_dev, size=scanner.num_beams)
            st.scans[i] = np.clip(noisy, scanner.min_range, scanner.max_range)

    return update_scans


def wall_gap(wall_scans, side_distances):
    """Per car: gap (m) from its body to the nearest wall in any direction."""
    return np.min(wall_scans - side_distances, axis=1)


def traffic_gap(env, cars):
    """For cars (indices): gap to the nearest other car in its path - beams
    within +-(width/2 + 0.5 m) of its heading that hit a car (more than 0.5 m
    shorter than the walls-only scan). inf when the path is clear."""
    gap = np.full(len(cars), np.inf)
    if env.num_agents < 2 or not len(cars):
        return gap
    now = env.unwrapped.sim._adjusted_scans[cars]
    in_path = (env.beam_cos > 0) & (np.abs(now * env.beam_sin) < env.path_half_width)
    car = in_path & (now < env.wall_scans[cars] - 0.5)
    rows = car.any(axis=1)
    if rows.any():
        d = np.where(car, np.maximum(now - env.side_distances[cars], 0.0), np.inf)
        gap[rows] = d[rows].min(axis=1)
    return gap


# -------------------------------------------------------------------------
# Pairwise geometry (once per step)
# -------------------------------------------------------------------------

def relative(xy, yaw):
    """(along, side)[i, j]: where point j is in car i's frame - metres ahead
    of it along its heading, and to its left."""
    d = xy[None, :, :] - xy[:, None, :]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c


class Geometry:
    """Every pairwise quantity between the racing cars, computed once per
    step and shared by the mirrors, the tow, the alongside inputs and the
    close calls. Arrays are over the racing cars only (idx)."""

    def __init__(self, cfg, poses, speed, racing):
        self.idx = np.flatnonzero(racing)
        self.count = n = len(self.idx)
        p = poses[self.idx].astype(np.float64)
        self.xy, self.yaw, self.v = p[:, :2], p[:, 2], speed[self.idx].astype(np.float64)
        if n < 2:
            return
        veh = cfg.vehicle
        self.along, self.side = relative(self.xy, self.yaw)              # rear axles
        heading = np.stack([np.cos(self.yaw), np.sin(self.yaw)], axis=1)
        self.body = self.xy + veh.collision_body_center_x * heading
        self.along_b, self.side_b = relative(self.body, self.yaw)        # body centres
        self.dyaw = self.yaw[None, :] - self.yaw[:, None]                # [i, j] = yaw_j - yaw_i
        # Body-to-body gaps: each car as 3 circles of radius width/2 along its length.
        r = veh.width / 2
        offs = np.array([-(veh.length / 2 - r), 0.0, veh.length / 2 - r])
        circles = self.body[:, None, :] + offs[None, :, None] * heading[:, None, :]          # (n, 3, 2)
        diff = circles[:, None, :, None, :] - circles[None, :, None, :, :]
        d2 = np.einsum("ijabk,ijabk->ijab", diff, diff).reshape(n, n, 9).min(axis=2)
        self.body_gap = np.sqrt(d2) - 2 * r
        np.fill_diagonal(self.body_gap, np.inf)


def mirrors(cfg, geo, n):
    """Per car: the nearest racing car behind within mirror_range and
    +-mirror_width sideways: (distance / range, closing speed / 20, side / 5,
    + = left); (1, 0, 0) when there's nobody."""
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, 0] = 1.0
    if geo.count < 2:
        return out
    s = cfg.sensors
    along, side = geo.along, geo.side
    behind = (along < 0) & (along > -s.mirror_range) & (np.abs(side) < s.mirror_width)
    dist = np.where(behind, np.hypot(along, side), np.inf)
    j = dist.argmin(axis=1)
    k = np.arange(geo.count)
    seen = np.isfinite(dist[k, j])
    closing = geo.v[j] * np.cos(geo.dyaw[k, j]) - geo.v
    out[geo.idx, 0] = np.where(seen, dist[k, j] / s.mirror_range, 1.0)
    out[geo.idx, 1] = np.where(seen, np.clip(closing / 20.0, -1.0, 1.0), 0.0)
    out[geo.idx, 2] = np.where(seen, np.clip(side[k, j] / 5.0, -1.0, 1.0), 0.0)
    return out


def alongside(cfg, geo, n):
    """Per car, left then right: the nearest racing car whose body overlaps
    front to back with a sideways gap < alongside_range: (gap / range,
    lateral closing speed / 10); (1, 0) for a side with nobody."""
    out = np.tile(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32), (n, 1))
    if geo.count < 2:
        return out
    veh = cfg.vehicle
    rng = cfg.sensors.alongside_range
    gap = np.abs(geo.side_b) - veh.width
    lateral = geo.v[None, :] * np.sin(geo.dyaw)                  # j's sideways speed in i's frame (+ = left)
    beside = (np.abs(geo.along_b) < veh.length) & (gap < rng)
    np.fill_diagonal(beside, False)
    k = np.arange(geo.count)
    for col, sign in ((0, 1.0), (2, -1.0)):
        g = np.where(beside & (geo.side_b * sign > 0), gap, np.inf)
        j = g.argmin(axis=1)
        seen = np.isfinite(g[k, j])
        closing = -sign * lateral[k, j]
        out[geo.idx, col] = np.where(seen, np.clip(g[k, j] / rng, 0.0, 1.0), 1.0)
        out[geo.idx, col + 1] = np.where(seen, np.clip(closing / 10.0, -1.0, 1.0), 0.0)
    return out


def track_preview(cfg, tracker, s, yaw, racing):
    """Per car: the track heading at s + track_preview_m minus the car's
    heading, wrapped, / pi - it knows the corners coming, also when cars
    block the lidar."""
    out = np.zeros((len(s), len(cfg.sensors.track_preview_m)), dtype=np.float32)
    idx = np.flatnonzero(racing)
    if len(idx):
        ahead = s[idx, None] + np.asarray(cfg.sensors.track_preview_m)[None, :]
        rel = tracker.yaw_at(ahead) - yaw[idx, None]
        out[idx] = ((rel + np.pi) % (2 * np.pi) - np.pi) / np.pi
    return out


# -------------------------------------------------------------------------
# Observation
# -------------------------------------------------------------------------

def build_obs(cfg, scans, prev_scans, std, yaw_rate, prev_action, race_pos, other_cars, drs, ers, mirror,
              preview, beside, design):
    """Observation vectors (cars, obs_dim), everything scaled to about
    [-1, 1] (layout: config.ObsLayout). design: each car's levels as
    design.inputs() (stock = 0)."""
    L = cfg.obs
    out = np.empty((len(scans), L.dim), dtype=np.float32)
    beams = cfg.obs_beams
    scan = scans[:, beams].astype(np.float32)
    prev = prev_scans[:, beams].astype(np.float32)
    out[:, L.lidar] = np.sqrt(np.clip(scan / cfg.sensors.lidar_range, 0.0, 1.0))
    # Closing speed in units of top speed: +1 = the gap shrinks as fast as
    # driving at top speed straight at it.
    out[:, L.closing] = np.clip((prev - scan) / (cfg.sim.control_dt * cfg.v_max), -2.0, 2.0)
    k = L.state.start
    out[:, k] = std[:, SV] / cfg.v_max
    out[:, k + 1] = std[:, SDELTA] / cfg.vehicle.s_max
    out[:, k + 2] = np.clip(yaw_rate / 2.0, -2.0, 2.0)
    out[:, k + 3] = np.clip(std[:, SBETA], -1.0, 1.0)
    out[:, k + 4:k + 7] = prev_action[:, :3]                     # previous steer (share of the available steering), throttle, brake
    out[:, L.race.start] = race_pos
    out[:, L.race.start + 1] = min(other_cars, cfg.sensors.max_cars_obs) / cfg.sensors.max_cars_obs
    out[:, L.drs] = drs
    out[:, L.ers] = ers
    out[:, L.mirrors] = mirror
    out[:, L.preview] = preview
    out[:, L.alongside] = beside
    out[:, L.design] = design
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)
