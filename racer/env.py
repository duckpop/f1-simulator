"""
RaceEnv: up to 22 cars on one track. Normalized actions in, numpy arrays
out:

    obs        (cars, obs_dim) float32
    reward     (cars,)
    terminated (cars,) bool - crash or no progress (a real terminal)
    truncated  (cars,) bool - the race timer ran out (bootstraps)

Every car is paid by its own profile's weights, so drivers of different
styles share a race, and every car has its own design (set_designs; the
stock car by default): its physics parameters and the design inputs of its
observation. step() returns the actions actually used (after the launch
override) as info["actions"]; those are what training stores.
"""

import dataclasses
import types

import numpy as np

from racer import design, physics, sensors
from racer.config import A_BRAKE, A_DRS, A_ERS, A_STEER, A_THROTTLE
from racer.track import SDELTA, SV, SYAW, CenterlineTracker, make_gym_env, patch_sim, spawn_states, unpatch_sim

REASONS = ("timeout", "collision", "contact", "no_progress", "sim_error")


class RaceEnv:
    def __init__(self, cfg, num_agents, map_name, profiles, render=False, seed=12345):
        self.cfg = cfg
        self.map_name = map_name
        self.num_agents = n = num_agents
        self.env = make_gym_env(cfg, n, map_name, render=render, seed=seed)
        self.unwrapped = self.env.unwrapped
        self.render_enabled = render
        sim = self.unwrapped.sim
        self.max_steps = cfg.episode_steps
        self.tracker = CenterlineTracker(self.unwrapped.track, n)
        self.tracker.build_drs_zones(cfg.drs.zone_max_turn_deg, cfg.drs.zone_lookahead_m, cfg.drs.min_zone_length_m)
        self.detect_s = self.tracker.final_corner_exit(cfg.race.detection_turn_deg)     # DRS / ERS detection point
        self.done = np.zeros(n, dtype=bool)
        self.wall_hit = np.zeros(n, dtype=bool)                        # set by the scan update
        self.wall_scans = np.full((n, cfg.sensors.lidar_beams), float(cfg.sensors.lidar_range))
        patch_sim(sim, self)
        self.side_distances = np.stack([c.side_distances for c in sim.scan_cache]).astype(np.float64)
        angles = sim.scan_cache[0].angles.astype(np.float64)
        self.beam_cos, self.beam_sin = np.cos(angles), np.sin(angles)
        self.path_half_width = cfg.vehicle.width / 2 + 0.5             # the path a car is about to drive
        self.base_params = sim.vehicle_params
        self.car_variation = (1.0, 1.0)                                 # (grip, mass) this episode
        self.pace = np.ones(n)
        # Fastest time through each sector on this track, across episodes (a
        # worker hands every training env of a track the same array).
        self.sector_record = np.full(cfg.race.lap_sectors, np.inf)
        self.set_profiles(profiles)
        self.set_designs(None)

    def set_profiles(self, profiles):
        """One profile name for every car, or one per car (cycled). Builds the
        per-car reward weights and resets every car's top speed (workers slow
        the opponents down per episode)."""
        names = [profiles] if isinstance(profiles, str) else list(profiles)
        self.profile_names = [names[i % len(names)] for i in range(self.num_agents)]
        rewards = [self.cfg.profile(name).rewards for name in self.profile_names]
        self.w = types.SimpleNamespace(**{k: np.array([r[k] for r in rewards], dtype=np.float64)
                                          for k in rewards[0].keys()})
        self.speed_share = np.ones(self.num_agents)          # share of its top speed each car runs at

    def set_designs(self, designs):
        """None (every car stock), one design for every car, or one per car
        (refused if illegal). Builds the per-car physics parameters, their
        rows in the integrator and the design inputs of the observation."""
        self.designs = design.per_car(self.cfg, designs, self.num_agents)
        self.car = design.car_params(self.cfg, self.designs)
        self.design_obs = design.inputs(self.cfg, self.designs)
        self.param_rows = physics.param_rows(self.cfg, self.unwrapped.sim.params_array, self.car)

    # ---- state access ---------------------------------------------------------
    @property
    def std(self):
        return self.unwrapped.sim.state.standard_state

    def race_distance(self):
        return self.race_offsets + self.progress

    def position_scores(self):
        """What places are ranked by: race distance on a grid or rolling start,
        distance driven on a random spawn (every car starts level)."""
        return self.race_distance() if self.spawn in ("grid", "rolling") else self.progress

    def avg_speed(self):
        return self.speed_sum / np.maximum(self.racing_steps, 1)

    def laps(self):
        return self.progress / self.tracker.length

    def best_laps(self):
        return [min(t) if t else None for t in self.lap_times]

    # ---- reset -----------------------------------------------------------------
    def reset(self, rng, spawn="random", vary_car=False):
        """spawn: grid, rolling or random. vary_car (training only): this
        episode's grip / mass variation and each car's pace."""
        cfg, n = self.cfg, self.num_agents
        self._set_car_variation(rng if vary_car else None)
        self.pace = 1.0 - rng.uniform(0.0, cfg.variation.pace_spread, n) if vary_car else np.ones(n)
        states, self.race_offsets = spawn_states(self.tracker, cfg, n, rng, spawn)
        self.spawn = spawn
        self.launch_steps = cfg.launch_steps if spawn == "grid" else 0
        self.env.reset(options={"states": states})
        self.done = np.zeros(n, dtype=bool)          # before the scan: only racing cars are scanned
        self.wall_hit[:] = False
        cap = self.car.ers_capacity
        self.battery = cfg.ers.start_charge * cap if spawn == "grid" else rng.uniform(0.2, 1.0, n) * cap
        # The gym casts lidar only inside step(): cast it now so the first observation is real.
        self.unwrapped.sim._update_scans()
        self.prev_scans = self.unwrapped.sim.state.scans.copy()

        self.steps = 0
        self.prev_action = np.zeros((n, cfg.action_dim), dtype=np.float32)
        self.s = np.array([self.tracker.reset_agent(i, states[i, 0], states[i, 1]) for i in range(n)])
        self.progress = np.zeros(n)
        self.best_progress = np.zeros(n)
        self.steps_since_best = np.zeros(n, dtype=np.int64)
        self.max_speed = np.zeros(n)
        self.speed_sum = np.zeros(n)
        self.racing_steps = np.zeros(n, dtype=np.int64)
        self.tow_steps = np.zeros(n, dtype=np.int64)
        self.close_calls = np.zeros(n, dtype=np.int64)
        self.drs_steps = np.zeros(n, dtype=np.int64)
        self.ers_steps = np.zeros(n, dtype=np.int64)
        self.battery_sum = np.zeros(n)
        self.drs_open = np.zeros(n)
        self.ers_on = np.zeros(n)
        # Detection point (the final corner's exit), from the end of lap 1: passing
        # it less than the gap behind the car that passed before gives the chase
        # recharge (ERS) and arms DRS until the next pass.
        self.ers_close = np.zeros(n, dtype=bool)
        self.drs_armed = np.zeros(n, dtype=bool)
        self.detect_gap = np.full(n, np.nan)        # s behind the previous car at the last counted pass
        self.last_detection = -np.inf               # when any car last passed the detection point
        self.prev_body_gap = np.full((n, n), np.inf)
        self.prev_wall_gap = sensors.wall_gap(self.wall_scans, self.side_distances)
        # Lap timing: line crossing to line crossing, interpolated in the step.
        self.lap_start = np.full(n, np.nan)
        self.lap_times = [[] for _ in range(n)]
        self.sector_len = self.tracker.length / cfg.race.lap_sectors
        self.sector_idx = (self.s // self.sector_len).astype(np.int64) % cfg.race.lap_sectors
        self.sector_entry = np.full(n, np.nan)
        self.best_sector = np.full(cfg.race.lap_sectors, np.inf)      # this race's best per sector
        self.purple_sectors = np.zeros(n, dtype=np.int64)
        self.termination_reason = [None] * n
        # ahead[i, j]: car i ahead of car j by race distance (passes are counted
        # on it); place_ahead: the same by position score (race points).
        self.ahead = self.race_offsets[:, None] > self.race_offsets[None, :]
        score = self.position_scores()
        self.place_ahead = score[:, None] > score[None, :]
        racing = ~self.done
        self.points = self._points(racing)
        self.start_points = self.points.copy()
        self.place_paid = np.zeros(n)                # race-point payments so far (before weights)
        self.overtakes = np.zeros(n, dtype=np.int64)
        self.times_overtaken = np.zeros(n, dtype=np.int64)
        self.pass_step = np.full((n, n), -1, dtype=np.int64)          # [i, j]: step i got past j, pending
        self.passes_on = np.zeros((n, n), dtype=np.int64)            # [i, j]: i passed j (paid)
        self.gain_ref = np.full(n, -1, dtype=np.int64)
        self.gain_gap = np.zeros(n)
        self.finish_position = [None] * n
        self.geo = sensors.Geometry(cfg, self.unwrapped.sim.state.poses, self.std[:, SV], racing)
        self._update_drs_available(racing)
        return self._observe(racing)

    # ---- step ------------------------------------------------------------------
    def step(self, actions):
        cfg, n = self.cfg, self.num_agents
        a = np.clip(np.asarray(actions, dtype=np.float32).reshape(n, cfg.action_dim), -1.0, 1.0)
        if self.steps < self.launch_steps:                   # lights out: everyone launches flat out
            a[:, A_THROTTLE] = 1.0
            a[:, A_BRAKE] = np.minimum(a[:, A_BRAKE], 0.0)
        active = ~self.done
        sim = self.unwrapped.sim
        car = self.car
        v0 = self.std[:, SV].astype(np.float64)
        throttle = np.maximum(a[:, A_THROTTLE], 0.0).astype(np.float64)
        brake = np.maximum(a[:, A_BRAKE], 0.0).astype(np.float64)
        # DRS opens only where available and closes the moment the car brakes;
        # latched, an open flap also stays open without the DRS output until then.
        pressed = (a[:, A_DRS] > 0) | ((self.drs_open > 0) if cfg.drs.latch else False)
        self.drs_open = (pressed & (self.drs_available > 0) & (brake == 0) & active).astype(np.float64)
        # The battery recharges faster for a lap after crossing the line close behind another car.
        e = cfg.ers
        scale = np.where(self.ers_close & active, e.close_harvest_rate / e.harvest_rate, 1.0) if e.harvest_rate > 0 else 1.0
        self.ers_on = physics.ers_update(cfg, car, self.battery, throttle, brake, a[:, A_ERS] > 0, v0, active, scale)
        pace = self.speed_share * self.pace
        engine = physics.engine_accel(cfg, car, v0, pace, self.ers_on)
        # Grip limit on the steering: the design's grip and steering, an open
        # DRS flap costs downforce, and braking / accelerating use the tyre too.
        decel = physics.braking(cfg, car, throttle, brake)              # brakes + engine braking off the throttle
        lateral = physics.lateral_share(cfg, car, throttle * engine - decel)
        cap = physics.max_steer(cfg, v0, car.grip * self.car_variation[0], 1.0 - cfg.drs.grip_loss * self.drs_open,
                                car.steer, lateral)
        if cfg.grip.steering_scaled:                         # the output is a share of the steering available now
            steer = a[:, A_STEER] * cap
        else:
            steer = np.clip(a[:, A_STEER] * cfg.vehicle.s_max * car.steer, -cap, cap)
        boost = physics.tow_boost(cfg, self.geo, n)          # from the end of the last step: same poses
        accel = physics.longitudinal_accel(cfg, car, v0, throttle, decel, engine, pace, boost, self.drs_open, ~active)
        steer[~active] = 0.0
        control = np.stack([steer, accel], axis=1).astype(np.float32)
        self.tow_steps += boost > 0
        self.drs_steps += self.drs_open > 0
        self.ers_steps += self.ers_on > 0
        self.battery_sum += active * self.battery / car.ers_capacity

        reward = np.zeros(n, dtype=np.float32)
        terminated = np.zeros(n, dtype=bool)
        truncated = np.zeros(n, dtype=bool)
        self.prev_scans = sim.state.scans.copy()
        try:
            if self.render_enabled:
                self.env.step(control)        # also builds the renderer's per-car dicts
            else:
                sim.step(control)
            sim_ok = bool(np.all(np.isfinite(self.std[active])))
        except Exception:
            sim_ok = False
        if not sim_ok:
            # Numerical blow-up: every active car crashes and the race is over.
            reward[active] = self.w.wall_crash[active]
            terminated[active] = True
            for i in np.flatnonzero(active):
                self.termination_reason[i] = "sim_error"
            self.done[:] = True
            return np.zeros((n, cfg.obs_dim), dtype=np.float32), reward, terminated, truncated, {"actions": a}

        self.steps += 1
        st = sim.state
        # No reversing: an integrator substep can overshoot a hair below 0.
        backwards = st.state[:, 3] < 0
        if backwards.any():
            st.state[backwards, 3] = 0.0
            st.standard_state[backwards, SV] = 0.0
        std = self.std
        cars = np.flatnonzero(active)
        s_old = self.s[cars]
        s_new = self.tracker.update(cars, std[cars, :2].astype(np.float64))
        ds = self.tracker.delta(s_new, s_old)

        # Lap line and sector boundaries: the only per-car branch.
        sectors = cfg.race.lap_sectors
        k_new = (s_new // self.sector_len).astype(np.int64) % sectors
        purple = np.zeros(len(cars))
        pace = np.zeros(len(cars))
        record = np.zeros(len(cars))
        events = ((ds > 0) & (s_new < s_old)) | ((ds < 0) & (s_new > s_old)) | (k_new != self.sector_idx[cars])
        for m in np.flatnonzero(events):
            self._time_lap(cars[m], s_old[m], s_new[m], ds[m])
            purple[m], pace[m], record[m] = self._time_sector(cars[m], s_old[m], s_new[m], ds[m], k_new[m])
        # Passing the detection point (forwards), timed inside the step.
        to_point = (self.detect_s - s_old) % self.tracker.length
        passed = (ds > 0) & (to_point < ds)
        if passed.any():
            times = (self.steps - 1 + to_point[passed] / ds[passed]) * cfg.sim.control_dt
            self._detection_check(sorted(zip(times.tolist(), cars[passed].tolist())))

        self.s[cars] = s_new
        self.progress[cars] += ds
        speed = std[cars, SV].astype(np.float64)
        self.max_speed[cars] = np.maximum(self.max_speed[cars], speed)
        self.speed_sum[cars] += speed
        self.racing_steps[cars] += 1
        self.purple_sectors[cars] += purple > 0
        w = self.w
        r = ds * cfg.progress_scale * w.progress_weight[cars]
        r -= w.steer_change[cars] * np.abs(a[cars, A_STEER] - self.prev_action[cars, A_STEER])
        r += purple * w.fastest_sector_lap[cars] / sectors + pace * w.sector_pace[cars] + record * w.sector_record[cars]

        # No progress: the timer runs while the car isn't gaining ground, and
        # pauses while another car blocks its path (it's queuing, not lost).
        improved = self.progress[cars] > self.best_progress[cars] + 0.5
        self.best_progress[cars[improved]] = self.progress[cars[improved]]
        self.steps_since_best[cars[improved]] = 0
        stuck = cars[~improved]
        if len(stuck):
            clear = sensors.traffic_gap(self, stuck) > cfg.spawn.no_progress_blocked_gap
            self.steps_since_best[stuck[clear]] += 1

        crash = st.collisions[cars] > 0
        wall = crash & self.wall_hit[cars]
        contact = crash & ~self.wall_hit[cars]
        no_progress = ~crash & (self.steps_since_best[cars] >= cfg.no_progress_steps)
        timeout = ~crash & ~no_progress & (self.steps >= self.max_steps)
        r += (wall | no_progress) * w.wall_crash[cars] + contact * w.car_crash[cars]
        reward[cars] = r
        terminated[cars] = crash | no_progress
        truncated[cars] = timeout
        for flags, reason in ((wall, "collision"), (contact, "contact"), (no_progress, "no_progress"),
                              (timeout, "timeout")):
            for i in cars[flags]:
                self.termination_reason[i] = reason

        racing = active & ~terminated            # incl. cars whose time is up: the final state bootstraps
        self.geo = sensors.Geometry(cfg, st.poses, std[:, SV], racing)
        wall_close = self._wall_close_call(racing)
        car_close = np.zeros(n)
        if n > 1:
            reward += self._race_rewards(racing, time_up=self.steps >= self.max_steps)
            car_close = self._car_close_call(racing)
        reward -= np.maximum(w.wall_close_call * wall_close, w.car_close_call * car_close).astype(np.float32)
        self.close_calls += np.maximum(wall_close, car_close) > 0

        self.prev_action[active] = a[active]
        out_now = active & (terminated | truncated)
        self.done |= terminated | truncated
        self._update_drs_available(racing)
        obs = self._observe(racing)
        if out_now.any() and not self.done.all():
            self._clear_cars(np.flatnonzero(out_now))
        return obs, reward, terminated, truncated, {"actions": a}

    # ---- observation -----------------------------------------------------------
    def _observe(self, racing):
        cfg, n = self.cfg, self.num_agents
        std = self.std
        yaw_rate = std[:, SV] * np.tan(std[:, SDELTA]) / cfg.wheelbase       # KS reports 0
        drs = np.stack([self.drs_available, self.drs_open, self.drs_zone_left / 500.0], axis=1)
        ers = np.stack([self.battery / self.car.ers_capacity, self.ers_on], axis=1)
        return sensors.build_obs(
            cfg, self.unwrapped.sim.state.scans, self.prev_scans, std, yaw_rate, self.prev_action,
            self._position_terms(racing), n - 1, drs, ers, sensors.mirrors(cfg, self.geo, n),
            sensors.track_preview(cfg, self.tracker, self.s, std[:, SYAW].astype(np.float64), racing),
            sensors.alongside(cfg, self.geo, n), self.design_obs)

    def _position_terms(self, racing):
        """(racing cars behind - racing cars ahead) / (cars - 1): +1 leading, -1 last."""
        n = self.num_agents
        out = np.zeros(n)
        idx = np.flatnonzero(racing)
        if n > 1 and len(idx) > 1:
            sc = self.position_scores()[idx]
            out[idx] = ((sc[:, None] > sc[None, :]).sum(axis=1) - (sc[:, None] < sc[None, :]).sum(axis=1)) / (n - 1)
        return out

    # ---- DRS and the detection point ------------------------------------------------
    def _detection_check(self, passes):
        """Passes of the detection point (the final corner's exit) this step,
        [(time, car)] in order. From [race] detection_from_s on (the end of lap
        1), a car passing less than [ers] close_gap_s after the previous car -
        any car, a backmarker too - gets the chase recharge until its next
        pass, and less than [drs] gap_s arms DRS until then."""
        cfg = self.cfg
        for t, i in passes:
            gap, self.last_detection = t - self.last_detection, t
            if t < cfg.race.detection_from_s:
                continue
            self.detect_gap[i] = gap
            self.ers_close[i] = gap < cfg.ers.close_gap_s
            self.drs_armed[i] = gap < cfg.drs.gap_s

    def _update_drs_available(self, racing):
        """Available in a DRS zone, after drs.from_s, above drs.min_speed and
        (gap_s > 0) armed at the detection point (passed it less than gap_s
        behind the previous car)."""
        cfg, d = self.cfg, self.cfg.drs
        self.drs_zone_left = np.where(racing, self.tracker.zone_left(self.s), 0.0)
        avail = np.zeros(self.num_agents)
        if self.steps * cfg.sim.control_dt >= d.from_s:
            v = self.std[:, SV].astype(np.float64)
            ok = racing & (self.drs_zone_left > 0) & (v >= d.min_speed)
            if d.gap_s > 0:
                ok &= self.drs_armed
            avail = ok.astype(np.float64)
        self.drs_available = avail

    # ---- laps and sectors --------------------------------------------------------
    def _time_lap(self, i, s_old, s_new, ds):
        """A lap runs from one forward line crossing to the next, interpolated
        inside the step; crossing backwards voids the lap in progress."""
        dt = self.cfg.sim.control_dt
        if ds > 0 and s_new < s_old:
            frac = min(max((self.tracker.length - s_old) / ds, 0.0), 1.0)
            crossed_at = (self.steps - 1 + frac) * dt
            if np.isfinite(self.lap_start[i]):
                self.lap_times[i].append(float(crossed_at - self.lap_start[i]))
            self.lap_start[i] = crossed_at
        elif ds < 0 and s_new > s_old:
            self.lap_start[i] = np.nan

    def _time_sector(self, i, s_old, s_new, ds, k_new):
        """When car i completes a sector: (purple - the race's fastest time
        through it so far, pace - 0..1 how close to the track's fastest time
        through it, record - beat it). The first time through a sector only
        sets its reference; a sector partly driven backwards isn't timed."""
        cfg = self.cfg
        sectors = cfg.race.lap_sectors
        k_old = self.sector_idx[i]
        if k_new == k_old:
            return 0.0, 0.0, 0.0
        self.sector_idx[i] = k_new
        if ds <= 0 or k_new != (k_old + 1) % sectors:
            self.sector_entry[i] = np.nan
            return 0.0, 0.0, 0.0
        frac = min(max(((k_new * self.sector_len - s_old) % self.tracker.length) / ds, 0.0), 1.0)
        crossed_at = (self.steps - 1 + frac) * cfg.sim.control_dt
        entered, self.sector_entry[i] = self.sector_entry[i], crossed_at
        if not np.isfinite(entered):
            return 0.0, 0.0, 0.0
        t = crossed_at - entered
        best = self.best_sector[k_old]
        purple = float(t < best and np.isfinite(best))
        self.best_sector[k_old] = min(best, t)
        ref = self.sector_record[k_old]
        pace = record = 0.0
        if np.isfinite(ref):
            pace = float(np.clip(1.0 - (t / ref - 1.0) / cfg.race.sector_pace_window, 0.0, 1.0))
            record = float(t < ref)
        self.sector_record[k_old] = min(ref, t)
        return purple, pace, record

    # ---- close calls -------------------------------------------------------------
    def _contact_level(self, gap, closing):
        """0..1: how close a gap shrinking at `closing` m/s is to 0 within
        close_call_ttc; 0 unless it shrinks faster than close_call_min_closing."""
        rc = self.cfg.race
        closing = np.where(np.isfinite(closing) & (closing > rc.close_call_min_closing), closing, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing > 0, np.maximum(gap, 0.0) / np.maximum(closing, 1e-6), np.inf)
        return np.clip(1.0 - ttc / rc.close_call_ttc, 0.0, 1.0)

    def _car_close_call(self, racing):
        """Per car: the highest level over the other cars, from body-to-body
        gaps (both cars of a pair pay: either can make room)."""
        n = self.num_agents
        gap = np.full((n, n), np.inf)
        if self.geo.count > 1:
            gap[np.ix_(self.geo.idx, self.geo.idx)] = self.geo.body_gap
        with np.errstate(invalid="ignore"):                    # inf - inf: a pair just appeared
            closing = (self.prev_body_gap - gap) / self.cfg.sim.control_dt
        self.prev_body_gap = gap
        return self._contact_level(gap, closing).max(axis=1) * racing

    def _wall_close_call(self, racing):
        """Per car: from the nearest-wall gap and how fast it shrinks (the
        nearest wall only: per-beam TTC fired on 47-85% of clean steps at
        80 m/s - every corner's outside wall is "0.4 s ahead")."""
        gap = sensors.wall_gap(self.wall_scans, self.side_distances)
        with np.errstate(invalid="ignore"):
            level = self._contact_level(gap, (self.prev_wall_gap - gap) / self.cfg.sim.control_dt) * racing
        self.prev_wall_gap = gap
        return level

    # ---- race order, passes, points ----------------------------------------------
    def _race_rewards(self, racing, time_up):
        """Passes (held pass_hold_s), gain on the car ahead, race points as they
        change; records the finishing order when the timer runs out."""
        cfg, n = self.cfg, self.num_agents
        w = self.w
        bonus = np.zeros(n)
        # Every pair (parked cars too), so passing a wreck never counts later.
        passed = self._update_order(self.ahead, self.race_distance())        # [i, j]: i just got past j
        self._update_order(self.place_ahead, self.position_scores())
        both = racing[:, None] & racing[None, :]
        # A pass pays once held with both cars racing; a crash of either car
        # or a retake cancels it, and a retake isn't a pass of its own.
        new = passed & both
        retaken = new & (self.pass_step.T >= 0)
        self.pass_step[retaken.T] = -1
        self.pass_step[new & ~retaken] = self.steps
        pending = self.pass_step >= 0
        self.pass_step[pending & (self.ahead.T | ~both)] = -1
        confirmed = (self.pass_step >= 0) & ((self.steps - self.pass_step >= cfg.pass_hold_steps) | time_up)
        if confirmed.any():
            opening = (self.spawn == "grid") & (self.pass_step * cfg.sim.control_dt <= cfg.race.start_phase_s)
            weight = np.where(opening, cfg.race.start_overtake_mult, 1.0) * confirmed
            bonus += w.overtake * weight.sum(axis=1) - w.overtaken * weight.sum(axis=0)
            self.passes_on += confirmed
            self.overtakes += confirmed.sum(axis=1)
            self.times_overtaken += confirmed.sum(axis=0)
            self.pass_step[confirmed] = -1

        bonus += w.gain_per_m * self._gain_on_car_ahead(racing)

        points = self._points(racing)
        self.place_paid += points - self.points
        bonus += w.per_place * (points - self.points)
        self.points = points

        if time_up and racing.any():
            score = self.position_scores()
            for rank, i in enumerate(sorted(np.flatnonzero(racing), key=lambda i: -score[i])):
                self.finish_position[i] = rank + 1
        return bonus.astype(np.float32)

    def _update_order(self, order, score):
        """order[i, j]: i ahead of j. A pair flips once the car behind leads
        by more than overtake_margin (no flicker side by side); returns the flips."""
        flips = ~order & (score[:, None] - score[None, :] > self.cfg.race.overtake_margin)
        order |= flips
        order &= ~flips.T
        return flips

    def _points(self, racing):
        """Race points each car holds: cars - racing cars ahead of it, 0 once
        out (the eval's place points)."""
        ahead = (self.place_ahead & racing[:, None]).sum(axis=0)
        return np.where(racing, self.num_agents - ahead, 0)

    def _gain_on_car_ahead(self, racing):
        """Metres each racing car closed on the nearest racing car ahead on
        the centerline within gain_range, if it's the same car as last step
        (metres lost cost the same; dropping back and catching up nets 0)."""
        n = self.num_agents
        gain = np.zeros(n)
        ref, gap = np.full(n, -1, dtype=np.int64), np.zeros(n)
        idx = np.flatnonzero(racing)
        if len(idx) > 1:
            s = self.s[idx]
            ahead = (s[None, :] - s[:, None]) % self.tracker.length
            np.fill_diagonal(ahead, np.inf)
            nearest = ahead.argmin(axis=1)
            dist = ahead[np.arange(len(idx)), nearest]
            close = dist < self.cfg.race.gain_range
            ref[idx[close]] = idx[nearest[close]]
            gap[idx[close]] = dist[close]
            same = (ref >= 0) & (ref == self.gain_ref)
            gain[same] = self.gain_gap[same] - gap[same]
        self.gain_ref, self.gain_gap = ref, gap
        return gain

    # ---- cars out of the race, car variation ------------------------------------
    def _clear_cars(self, idx):
        """Take cars that are out off the track: parked far outside the map, at
        rest - never scanned, ray-cast against or integrated again."""
        st = self.unwrapped.sim.state
        x, y = sensors.CLEARED_POSE
        st.state[idx] = 0.0
        st.state[idx, 0], st.state[idx, 1] = x, y
        st.standard_state[idx] = 0.0
        st.standard_state[idx, 0], st.standard_state[idx, 1] = x, y
        st.poses[idx] = (x, y, 0.0)

    def _set_car_variation(self, rng):
        """Grip and mass (+ yaw inertia) x U(1 +- variation) for the episode, the
        same for every car; rng=None restores the normal car. (The grip acts
        through the grip limit; the kinematic model ignores mass.)"""
        g = m = 1.0
        v = self.cfg.variation
        if rng is not None and (v.grip > 0 or v.mass > 0):
            g = 1.0 + rng.uniform(-v.grip, v.grip)
            m = 1.0 + rng.uniform(-v.mass, v.mass)
        if (g, m) == self.car_variation:
            return
        p = self.base_params
        params = p if (g, m) == (1.0, 1.0) else dataclasses.replace(p, mu=p.mu * g, m=p.m * m, I=p.I * m)
        sim = self.unwrapped.sim
        sim.update_params(params)
        self.param_rows = physics.param_rows(self.cfg, sim.params_array, self.car)
        self.car_variation = (g, m)

    def render(self):
        self.env.render()

    def close(self):
        # The patched methods reference this env (a reference loop), and the
        # gym env has loops of its own: callers that rebuild envs must
        # gc.collect() after closing, or closed tracks pile up (~60 MB each).
        unpatch_sim(self.unwrapped.sim)
        self.env.close()
