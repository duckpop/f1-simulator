"""
Tracks: the map list, building the gym env (and installing this project's
patches on its simulator), the centerline tracker (progress, lap line, track
headings, grid anchor, straights, DRS zones) and the spawn states.
"""

import dataclasses
import types

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionType
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.env_config import ControlConfig, EnvConfig, LoopCounterMode, SimulationConfig
from f1tenth_gym.envs.lidar import laser_models as _laser_models
from f1tenth_gym.envs.lidar.config import LiDARConfig

# std_state layout from the gym: [x, y, delta, v, yaw, yaw_rate, beta]
SX, SY, SDELTA, SV, SYAW, SYAWRATE, SBETA = range(7)

# --- One lidar distance map per track, shared by every car ------------------
# The gym gives every car its own lidar simulator, and each computes its own
# copy of the track's distance map (32 MB at scale 12): a 22-car track held
# ~1 GB. This computes it once per track bitmap and hands every car the same
# (read-only) array; at most 4 tracks per process.
_original_get_dt = _laser_models.get_dt
_dt_cache = {}


def _shared_get_dt(bitmap, resolution):
    key = (id(bitmap), bitmap.shape, float(resolution))
    hit = _dt_cache.get(key)
    if hit is not None and hit[0] is bitmap:
        return hit[1]
    dt = _original_get_dt(bitmap, resolution)
    if len(_dt_cache) >= 4:
        _dt_cache.pop(next(iter(_dt_cache)))
    _dt_cache[key] = (bitmap, dt)          # keeps the bitmap alive so its id can't be reused
    return dt


_laser_models.get_dt = _shared_get_dt


# -------------------------------------------------------------------------
# Maps and the gym env
# -------------------------------------------------------------------------

def available_maps(cfg):
    """{map name: path stem the gym loads} for every *_map.yaml under maps_dir
    with a matching *_centerline.csv. Folder and file names don't always
    match ("Mexico City/MexicoCity_map.yaml"). Falls back to the gym's
    built-in Spielberg when the folder is missing."""
    maps = {}
    for yaml_path in sorted(cfg.maps_dir.glob("*/*_map.yaml")):
        stem = yaml_path.name[:-len("_map.yaml")]
        if (yaml_path.parent / f"{stem}_centerline.csv").exists():
            maps[stem] = str(yaml_path.parent / stem)
    if not maps:
        maps["Spielberg"] = "Spielberg"
    return maps


def train_maps(cfg):
    maps = available_maps(cfg)
    if cfg.sim.train_maps == "all":
        return sorted(maps)
    missing = [m for m in cfg.sim.train_maps if m not in maps]
    if missing:
        raise ValueError(f"[sim] train_maps not found in {cfg.maps_dir}: {missing} (available: {', '.join(maps)})")
    return list(cfg.sim.train_maps)


def make_gym_env(cfg, num_agents, map_name, render=False, seed=12345):
    """The raw gym env: KS model, ACCL longitudinal mode (the env computes
    every car's acceleration), 20 Hz control integrated at 100 Hz, lidar
    simulated at exactly the beams used, and the gym's Frenet tracking off
    (it sticks at s ~ 1.8 m on scaled maps; CenterlineTracker replaces it)."""
    params = dataclasses.replace(cfg.vehicle, v_max=cfg.physics_v_max)
    gym_cfg = EnvConfig(
        params=params,
        map_name=available_maps(cfg)[map_name],
        map_scale=cfg.sim.map_scale,
        num_agents=num_agents,
        seed=seed,
        render_enabled=render,
        lidar_config=LiDARConfig(enabled=True, num_beams=cfg.sensors.lidar_beams,
                                 field_of_view=np.deg2rad(cfg.sensors.lidar_fov_deg),
                                 range_max=cfg.sensors.lidar_range, noise_std=cfg.sim.lidar_noise_std),
        control_config=ControlConfig(longitudinal_mode=LongitudinalActionType.ACCL),
        simulation_config=SimulationConfig(timestep=cfg.sim.control_dt, integrator_timestep=cfg.sim.integrator_dt,
                                           loop_counter=LoopCounterMode.TOGGLE, dynamics_model=DynamicModel.KS,
                                           compute_frenet_frame=False, max_laps=None),
    )
    return gym.make("f1tenth_gym:f1tenth-v0", config=gym_cfg, render_mode="human" if render else None)


def patch_sim(sim, env):
    """Install the racing-only scan update and physics step as bound methods
    of the gym's simulator (they read env.done and write env.wall_hit /
    env.wall_scans), plus smooth steering. unpatch_sim() removes them: they
    reference the env, i.e. a reference loop."""
    from racer import physics, sensors
    sim.steering_fn = physics.smooth_steering(sim.time_step)
    sim._update_scans = types.MethodType(sensors.racing_scan_update(env), sim)
    sim.step = types.MethodType(physics.racing_step(env), sim)


def unpatch_sim(sim):
    for name in ("_update_scans", "step"):
        sim.__dict__.pop(name, None)


# -------------------------------------------------------------------------
# Centerline
# -------------------------------------------------------------------------

class CenterlineTracker:
    """Arc-length position along the closed centerline for N cars, each
    found in a local window of segments around its last one (so it can't
    jump across the map), all cars at once."""

    GRID_STEP = 2.0      # m, resolution of the bend / DRS-zone grids

    def __init__(self, track, num_agents, window_back=5, window_fwd=20):
        cl = track.centerline
        pts = np.stack([np.asarray(cl.xs), np.asarray(cl.ys)], axis=1).astype(np.float64)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]                   # a closed loop stores its first point twice
        self.pts = pts
        self.seg = np.roll(pts, -1, axis=0) - pts
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.seg_len2 = np.maximum(self.seg_len ** 2, 1e-9)
        self.seg_yaw = np.arctan2(self.seg[:, 1], self.seg[:, 0])
        self.ss = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(np.sum(self.seg_len))
        self.n = len(pts)
        self.offsets = np.arange(-window_back, window_fwd + 1)
        self.idx = np.zeros(num_agents, dtype=np.int64)
        self._bend_cache = {}
        self.drs_left = np.zeros(1)          # set by build_drs_zones()

    def reset_agent(self, i, x, y):
        """Global projection - only at spawn."""
        xy = np.array([x, y])
        t = np.clip(np.einsum("ij,ij->i", xy - self.pts, self.seg) / self.seg_len2, 0.0, 1.0)
        dist = np.linalg.norm(xy - (self.pts + self.seg * t[:, None]), axis=1)
        k = int(np.argmin(dist))
        self.idx[i] = k
        return float(self.ss[k] + t[k] * self.seg_len[k])

    def update(self, cars, xy):
        """Arc lengths of cars (indices) at positions xy (cars, 2), each from
        a window of segments -5..+20 around its last one."""
        inds = (self.idx[cars, None] + self.offsets[None, :]) % self.n
        p, d = self.pts[inds], self.seg[inds]
        rel = xy[:, None, :] - p
        t = np.clip(np.einsum("kij,kij->ki", rel, d) / self.seg_len2[inds], 0.0, 1.0)
        off = rel - d * t[..., None]
        j = np.einsum("kij,kij->ki", off, off).argmin(axis=1)
        k = np.arange(len(cars))
        seg = inds[k, j]
        self.idx[cars] = seg
        return self.ss[seg] + t[k, j] * self.seg_len[seg]

    def delta(self, s_new, s_old):
        """Signed progress, unwrapped across the start/finish line (arrays)."""
        ds = np.asarray(s_new, dtype=np.float64) - s_old
        half = 0.5 * self.length
        return ds + self.length * (ds < -half) - self.length * (ds > half)

    def yaw_at(self, s):
        """Track heading (rad) at arc lengths s."""
        s = np.asarray(s, dtype=np.float64) % self.length
        i = np.clip(np.searchsorted(self.ss, s, side="right") - 1, 0, self.n - 1)
        return self.seg_yaw[i]

    def pose_at(self, s):
        s = float(s) % self.length
        i = int(np.clip(np.searchsorted(self.ss, s, side="right") - 1, 0, self.n - 1))
        t = (s - self.ss[i]) / max(self.seg_len[i], 1e-9)
        xy = self.pts[i] + t * self.seg[i]
        return float(xy[0]), float(xy[1]), float(self.seg_yaw[i])

    def _bend(self, depth, ahead):
        """(s, bend): for every point s along the lap (GRID_STEP apart), the
        largest heading change (rad) from `depth` m behind it to `ahead` m in
        front. Cached per (depth, ahead)."""
        key = (round(depth, 3), round(ahead, 3))
        if key not in self._bend_cache:
            step = self.GRID_STEP
            s_ext = np.arange(-self.length, 2 * self.length, step)
            yaw = np.unwrap(self.yaw_at(s_ext))
            first = int(round(self.length / step))
            cands = np.arange(first, first + int(self.length / step))          # s = 0 .. length
            win = np.arange(-int(np.ceil(depth / step)), int(np.ceil(ahead / step)) + 1)
            bend = np.abs(yaw[cands[:, None] + win[None, :]] - yaw[cands][:, None]).max(axis=1)
            self._bend_cache[key] = (s_ext[cands], bend)
        return self._bend_cache[key]

    def grid_anchor(self, depth, max_turn_deg, clear_ahead):
        """Arc length for pole: the point nearest the start line (either way)
        where the track stays within max_turn_deg from `depth` m behind it
        (the whole grid) to clear_ahead m in front of it (the launch).
        Falls back to the straightest spot."""
        s0, bend = self._bend(depth, clear_ahead)
        from_line = np.minimum(s0, self.length - s0)
        ok = bend <= np.deg2rad(max_turn_deg)
        best = int(np.argmin(np.where(ok, from_line, np.inf))) if ok.any() else int(np.argmin(bend))
        if from_line[best] <= self.GRID_STEP:
            return 0.0
        return float(s0[best]) if s0[best] <= self.length / 2 else float(s0[best] - self.length)

    def straight_spots(self, depth, ahead, max_turn_deg):
        """Every arc length where the track bends at most max_turn_deg from
        `depth` m behind to `ahead` m in front. Falls back to the straightest
        5% of the lap."""
        s0, bend = self._bend(depth, ahead)
        ok = bend <= np.deg2rad(max_turn_deg)
        if not ok.any():
            ok = bend <= np.quantile(bend, 0.05)
        return s0[ok]

    # ---- DRS / ERS detection point -----------------------------------------------
    def final_corner_exit(self, min_turn_deg, min_rate_deg_per_m=0.25, smooth_m=20.0):
        """Arc length of the final corner's exit - the DRS / ERS detection
        point. Corners are stretches where the heading turns faster than
        min_rate_deg_per_m (measured over +-smooth_m) and turns at least
        min_turn_deg in total; the final one is the last before the
        start/finish line. 0 (the line itself) if the track has none."""
        step = self.GRID_STEP
        s = np.arange(0.0, self.length, step)
        yaw = np.unwrap(self.yaw_at(s))
        k = max(1, int(round(smooth_m / step)))
        lap_turn = yaw[-1] + ((yaw[0] - yaw[-1] + np.pi) % (2 * np.pi) - np.pi) - yaw[0]   # ~ +-2 pi a lap
        ext = np.concatenate([yaw[-k:] - lap_turn, yaw, yaw[:k] + lap_turn])
        rate = np.abs(ext[2 * k:] - ext[:-2 * k]) / (2 * k * step)                       # rad per m
        turning = rate > np.radians(min_rate_deg_per_m)
        if turning.all() or not turning.any():
            return 0.0
        start = int(np.argmin(turning))                          # rotate so the array starts on a straight
        rot_turning, rot_rate = np.roll(turning, -start), np.roll(rate, -start)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], rot_turning.astype(np.int8), [0]])))
        exits = [s[(b - 1 + start) % len(s)] for a, b in zip(edges[0::2], edges[1::2])
                 if np.degrees(rot_rate[a:b].sum() * step) >= min_turn_deg]
        return float(max(exits)) if exits else 0.0

    # ---- DRS zones -----------------------------------------------------------
    def build_drs_zones(self, max_turn_deg, lookahead, min_length):
        """A point is in a DRS zone if the track heading stays within
        max_turn_deg of its own from there to `lookahead` m ahead, so every
        zone ends about one braking distance before the next corner. Zones
        shorter than min_length are dropped. Stores, per GRID_STEP point,
        the metres left to the zone's end (0 outside a zone)."""
        _, bend = self._bend(0.0, lookahead)
        ok = bend < np.deg2rad(max_turn_deg)
        step = self.GRID_STEP
        left = np.zeros(len(ok))
        if ok.all():
            left[:] = self.length                   # one endless straight
        elif ok.any():
            start = int(np.argmin(ok))              # rotate so the array starts outside a zone
            rot = np.roll(ok, -start)
            edges = np.flatnonzero(np.diff(np.concatenate([[0], rot.astype(np.int8), [0]])))
            for a, b in zip(edges[0::2], edges[1::2]):       # runs [a, b)
                if (b - a) * step >= min_length:
                    left[(np.arange(a, b) + start) % len(ok)] = (b - np.arange(a, b)) * step
        self.drs_left = left
        return left

    def zone_left(self, s):
        """Metres left in the DRS zone at arc lengths s (0 outside a zone)."""
        i = (np.asarray(s, dtype=np.float64) // self.GRID_STEP).astype(np.int64) % len(self.drs_left)
        return self.drs_left[i]

    def zones(self):
        """[(start s, length)] of the DRS zones, for printing."""
        left, step = self.drs_left, self.GRID_STEP
        starts = np.flatnonzero((left > 0) & (np.roll(left, 1) <= step))
        return [(float(i * step), float(left[i])) for i in starts]


# -------------------------------------------------------------------------
# Spawns
# -------------------------------------------------------------------------

def spawn_states(tracker, cfg, num_agents, rng, mode="random"):
    """Initial KS states [x, y, delta, v, yaw] and race offsets (each car's
    starting distance along the track relative to car 0).

    grid:    pole at grid_anchor, car i i*slot_gap behind it, 2 columns
             +-column_gap/2 off the centerline, standing still.
    rolling: a two-wide pack at speed on a straight with clear road ahead,
             random gaps and a random order.
    random:  spread evenly around the lap, small lateral / heading jitter,
             a random speed."""
    sp = cfg.spawn
    states = np.zeros((num_agents, 5), dtype=np.float32)
    offsets = np.zeros(num_agents)
    cols = 1 if num_agents == 1 else sp.grid_columns

    def place(i, s, lat, yaw_jitter=0.0, speed=0.0):
        x, y, yaw = tracker.pose_at(s)
        states[i] = (x - np.sin(yaw) * lat, y + np.cos(yaw) * lat, 0.0, speed, yaw + yaw_jitter)

    if mode == "grid":
        pole = tracker.grid_anchor((num_agents - 1) * sp.grid_slot_gap, sp.grid_max_turn_deg, sp.grid_clear_ahead)
        for i in range(num_agents):
            offsets[i] = -i * sp.grid_slot_gap
            place(i, pole + offsets[i], (i % cols - (cols - 1) / 2) * sp.grid_column_gap)
    elif mode == "rolling":
        offsets[1:] = -np.cumsum(rng.uniform(*sp.rolling_gap, size=max(num_agents - 1, 0)))
        spots = tracker.straight_spots(float(-offsets[-1]), sp.rolling_clear_ahead, sp.grid_max_turn_deg)
        front = float(rng.choice(spots))
        speed = rng.uniform(*sp.rolling_speed)
        for i in range(num_agents):
            lat = (i % cols - (cols - 1) / 2) * sp.grid_column_gap + rng.uniform(-0.5, 0.5)
            place(i, front + offsets[i], lat,
                  speed=speed * (1.0 + rng.uniform(-sp.rolling_speed_jitter, sp.rolling_speed_jitter)))
    elif mode == "random":
        base = rng.uniform(0.0, tracker.length)
        spacing = tracker.length / max(num_agents, 1)
        for i in range(num_agents):
            offsets[i] = i * spacing
            place(i, base + i * spacing, rng.uniform(-sp.random_lateral_jitter, sp.random_lateral_jitter),
                  yaw_jitter=np.deg2rad(rng.uniform(-sp.random_heading_jitter_deg, sp.random_heading_jitter_deg)),
                  speed=rng.uniform(*sp.random_speed))
    else:
        raise ValueError(f"unknown spawn mode {mode!r}")
    return states, offsets
