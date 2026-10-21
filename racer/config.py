"""
Settings: configs/settings.toml plus the driving-style profiles in
configs/profiles/*.toml, loaded into a read-only namespace (cfg.td3.batch,
cfg.rewards.overtake, ...) together with every constant derived from them:
the observation layout and lidar beam indices, the vehicle parameters, the
progress scale and the drag constant, and the designer's input / operation
sizes.

The Settings object is picklable (it re-derives itself from the raw tables),
so the learner resolves it once - CLI overrides included - and hands it to
every worker as an argument. Nothing in the project mutates module globals
to pass settings around.
"""

import dataclasses
import math
import os
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
SETTINGS_FILE = CONFIG_DIR / "settings.toml"
PROFILE_DIR = CONFIG_DIR / "profiles"

# Actions: steer, throttle, brake, DRS, ERS (all tanh outputs in [-1, 1]).
ACTION_NAMES = ("steer", "throttle", "brake", "drs", "ers")
ACTION_DIM = len(ACTION_NAMES)
A_STEER, A_THROTTLE, A_BRAKE, A_DRS, A_ERS = range(ACTION_DIM)


def _freeze(v):
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return {k: _freeze(x) for k, x in v.items()}
    return v


def _thaw(v):
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


class Section:
    """One TOML table as a read-only namespace."""

    __slots__ = ("_name", "_data")

    def __init__(self, name, data):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_data", {k: _freeze(v) for k, v in data.items()})

    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"no setting [{self._name}] {key}") from None

    def __setattr__(self, key, value):
        raise AttributeError("settings are read-only - pass overrides to config.load()")

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def as_dict(self):
        return dict(self._data)

    def __reduce__(self):
        return Section, (self._name, {k: _thaw(v) for k, v in self._data.items()})

    def __repr__(self):
        return f"[{self._name}] " + ", ".join(f"{k}={v!r}" for k, v in self._data.items())


@dataclasses.dataclass(frozen=True)
class Profile:
    """A driving style: its rewards and eval bonuses (base values + overrides)."""
    name: str
    description: str
    rewards: Section
    eval_bonus: Section


@dataclasses.dataclass(frozen=True)
class ObsLayout:
    """Where each block sits in the observation vector."""
    lidar: slice
    closing: slice
    state: slice
    race: slice
    drs: slice
    ers: slice
    mirrors: slice
    preview: slice
    alongside: slice
    design: slice
    dim: int


class Settings:
    """All settings, read-only. Sections are attributes (cfg.drs.boost); the
    derived constants are plain attributes too (cfg.obs_dim, cfg.drag, ...)."""

    def __init__(self, tables, profile_tables):
        object.__setattr__(self, "_tables", tables)
        object.__setattr__(self, "_profile_tables", profile_tables)
        for name, table in tables.items():
            object.__setattr__(self, name, Section(name, table))
        self._derive()
        self._profiles = {name: self._make_profile(name, t) for name, t in profile_tables.items()}
        self._validate()
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, key, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("settings are read-only - pass overrides to config.load()")
        object.__setattr__(self, key, value)

    def __reduce__(self):
        return Settings, (self._tables, self._profile_tables)

    # ---- derived constants -------------------------------------------------
    def _derive(self):
        from f1tenth_gym.envs.dynamic_models import FULLSCALE_VEHICLE_PARAMETERS
        fields = {f.name for f in dataclasses.fields(FULLSCALE_VEHICLE_PARAMETERS)}
        unknown = set(self.car.keys()) - fields
        if unknown:
            raise ValueError(f"[car] sets unknown vehicle parameters {sorted(unknown)}")
        self.vehicle = dataclasses.replace(FULLSCALE_VEHICLE_PARAMETERS, **self.car.as_dict())
        v = self.vehicle
        self.v_max = float(v.v_max)
        self.wheelbase = float(v.lf + v.lr)
        # Full throttle's power-limited acceleration equals the drag exactly at v_max.
        self.drag = float(v.a_max * v.v_switch / self.v_max ** 3)
        # Reward per metre of progress (x progress_weight): top speed for one step = +1.
        self.progress_scale = 1.0 / (self.v_max * self.sim.control_dt)
        # Top speed ~ power^(1/3) under v^2 drag.
        self.ers_speed_gain = (1.0 + self.ers.power_boost) ** (1.0 / 3.0) - 1.0
        # The physics' own speed ceiling sits above any tow + DRS + ERS speed of
        # the fastest legal design (+max_change); the real top speed comes from the drag.
        self.physics_v_max = (self.v_max * (1.0 + self.design.max_change) * (1.0 + self.tow.boost)
                              * (1.0 + self.drs.boost) * (1.0 + self.ers_speed_gain) * 1.05)

        dt = self.sim.control_dt
        if not math.isclose(dt / self.sim.integrator_dt, round(dt / self.sim.integrator_dt)):
            raise ValueError("[sim] control_dt must be a multiple of integrator_dt")
        self.episode_steps = int(self.sim.episode_steps)
        lo, hi = self.sim.train_race_s
        if not 0 < lo <= hi:
            raise ValueError("[sim] train_race_s must be [shortest, longest] seconds, 0 < shortest <= longest")
        self.train_race_steps = (int(round(lo / dt)), int(round(hi / dt)))
        self.launch_steps = int(round(self.spawn.launch_s / dt))
        self.no_progress_steps = int(round(self.spawn.no_progress_timeout_s / dt))
        self.pass_hold_steps = int(round(self.race.pass_hold_s / dt))

        self.obs_beams = self._obs_beam_indices()
        self.obs_beams.flags.writeable = False
        nb = len(self.obs_beams)
        sizes = [("lidar", nb), ("closing", nb), ("state", 7), ("race", 2), ("drs", 3), ("ers", 2),
                 ("mirrors", 3), ("preview", len(self.sensors.track_preview_m)), ("alongside", 4),
                 ("design", len(self.design.categories))]
        slices, k = {}, 0
        for name, size in sizes:
            slices[name] = slice(k, k + size)
            k += size
        self.obs = ObsLayout(**slices, dim=k)
        self.obs_dim = k
        self.action_dim = ACTION_DIM
        # Checkpoints carry this tag and are refused when it doesn't match.
        self.layout = f"racer-v2-{self.obs_dim}obs-{self.action_dim}act"

        # The designer (project_v2 §6.2): car levels + points used, each
        # driver's profile one-hot, the last race (7 per driver, team points,
        # "no race yet"), which operation this is. Operations: keep, add 1 to
        # a category, move 1 point from one category to another.
        categories = len(self.design.categories)
        self.designer_obs_dim = categories + 1 + 2 * len(self.season.profiles) + 2 * 7 + 2 + 1
        self.designer_ops = 1 + categories + categories * (categories - 1)
        self.designer_layout = f"designer-v1-{self.designer_obs_dim}obs-{self.designer_ops}ops"

        self.maps_dir = resolve(self.sim.maps_dir)
        self.save_dir = resolve(self.run.save_dir)
        self.worker_count = int(self.workers["count"]) if "count" in self.workers \
            else max(1, min(6, (os.cpu_count() or 4) - 4))

    def _obs_beam_indices(self):
        """The simulated beams the network gets ([sensors] obs_bands):
        symmetric around straight ahead, every beam near the centre and every
        2nd / 4th towards the sides."""
        n = int(self.sensors.lidar_beams)
        step = self.sensors.lidar_fov_deg / (n - 1)
        pos = np.arange(n) - (n - 1) / 2
        keep = []
        for i in range(n):
            angle = abs(pos[i]) * step
            for limit, every in self.sensors.obs_bands:
                if angle <= limit + 1e-9:
                    if round(abs(pos[i]) - (n % 2 == 0) * 0.5) % int(every) == 0:
                        keep.append(i)
                    break
        return np.array(keep, dtype=np.int64)

    # ---- validation -----------------------------------------------------------
    def _validate(self):
        from racer.design import CATEGORIES, check
        g = self.grip
        for key in ("steering_scaled", "traction_circle"):
            if not isinstance(g[key], bool):
                raise ValueError(f"[grip] {key} must be true or false")
        if not 0.0 <= g.combined_floor <= 1.0:
            raise ValueError("[grip] combined_floor must be within 0..1")
        if len(self.sensors.track_preview_m) < 1:
            raise ValueError("[sensors] track_preview_m needs at least one distance")

        d = self.design
        if tuple(d.categories) != CATEGORIES:
            raise ValueError(f"[design] categories must be {list(CATEGORIES)} (racer/design.py knows their effects)")
        lo, hi = d.levels
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo < hi):
            raise ValueError("[design] levels must be two integers [lowest, highest], 1 <= lowest < highest")
        if len(d.stock) != len(CATEGORIES) or any(not isinstance(x, int) or not lo <= x <= hi for x in d.stock):
            raise ValueError(f"[design] stock must be {len(CATEGORIES)} integer levels within {lo}-{hi}")
        if d.max_points < sum(d.stock):
            raise ValueError(f"[design] max_points ({d.max_points}) must be >= the stock car's {sum(d.stock)} points")
        check(self, d.stock, "[design] stock")
        if not 0.0 < d.step or not 0.0 <= d.max_change < 1.0 or d.step * (hi - lo) / 2 >= 1.0:
            raise ValueError("[design] needs step > 0, 0 <= max_change < 1 and every factor positive")
        if not 0.0 <= d.train_stock_share <= 1.0:
            raise ValueError("[design] train_stock_share must be within 0..1")

        s = self.season
        if not (isinstance(s.teams, int) and s.teams >= 1 and isinstance(s.races, int) and s.races >= 1
                and isinstance(s.max_moves, int) and s.max_moves >= 1):
            raise ValueError("[season] teams, races and max_moves must be integers >= 1")
        if 2 * s.teams > self.sensors.max_cars_obs:
            raise ValueError(f"[season] {s.teams} teams = {2 * s.teams} cars, more than "
                             f"[sensors] max_cars_obs ({self.sensors.max_cars_obs})")
        if not s.profiles or len(set(s.profiles)) != len(s.profiles):
            raise ValueError("[season] profiles must list distinct driving styles")
        for name in s.profiles:
            self.profile(name)

        if not 0.0 <= self.engine.brake_share <= 1.0 or self.engine.brake_limit < 1.0:
            raise ValueError("[engine] brake_share must be within 0..1 and brake_limit >= 1")
        if self.engine.brake_max <= 0.0:
            raise ValueError("[engine] brake_max (full braking) must be > 0")

        t = self.td3
        if t.replay_size < t.batch or t.host_replay_size < 0 or t.host_replay_margin_gb < 0:
            raise ValueError("[td3] needs replay_size >= batch, host_replay_size >= 0 and host_replay_margin_gb >= 0")

        dq = self.designer
        if not 0.0 <= dq.gamma < 1.0:
            raise ValueError("[designer] gamma must be within 0..1 (exclusive)")
        if len(dq.epsilon) != 2 or not 0.0 <= dq.epsilon[1] <= dq.epsilon[0] <= 1.0:
            raise ValueError("[designer] epsilon must be [start, end] with 0 <= end <= start <= 1")
        if dq.test_seasons < 1 or dq.workers < 1:
            raise ValueError("[designer] workers and test_seasons must be >= 1")

    # ---- profiles -----------------------------------------------------------
    def _make_profile(self, name, table):
        unknown = set(table) - {"description", "rewards", "eval_bonus"}
        if unknown:
            raise ValueError(f"profile {name!r}: unknown keys {sorted(unknown)} "
                             f"(a profile holds description, [rewards] and [eval_bonus])")
        merged = {}
        for section in ("rewards", "eval_bonus"):
            over = table.get(section, {})
            bad = set(over) - set(getattr(self, section).keys())
            if bad:
                raise ValueError(f"profile {name!r}: unknown [{section}] keys {sorted(bad)}")
            merged[section] = Section(section, {**getattr(self, section).as_dict(), **over})
        if merged["rewards"].overtaken < merged["rewards"].overtake:
            raise ValueError(f"profile {name!r}: overtaken must be >= overtake "
                             f"(otherwise two cars farm points by swapping places)")
        return Profile(name, str(table.get("description", "")), merged["rewards"], merged["eval_bonus"])

    @property
    def profile_names(self):
        return list(self._profiles)

    def profile(self, name):
        """The merged rewards + eval bonuses of a driving style."""
        try:
            return self._profiles[name]
        except KeyError:
            raise ValueError(f"unknown profile {name!r} - choose from {', '.join(self._profiles)} "
                             f"(or add configs/profiles/{name}.toml)") from None

    def color(self, profile):
        return self.colors[profile] if profile in self.colors else "#888888"

    # ---- paths ---------------------------------------------------------------
    def profile_dir(self, profile):
        return self.save_dir / profile

    def checkpoint_path(self, profile, tag):
        """A profile's checkpoint file: tag best, latest, an update count, ..."""
        return self.profile_dir(profile) / f"td3_{tag}.pt"

    @property
    def designer_dir(self):
        return self.save_dir / "designer"

    def designer_path(self, tag):
        """The designer's checkpoint file: tag best, latest, an update count."""
        return self.designer_dir / f"designer_{tag}.pt"

    # ---- overrides -------------------------------------------------------------
    def override(self, values):
        """A copy with {"section.key": value} replaced (CLI options)."""
        tables = {name: dict(t) for name, t in self._tables.items()}
        for dotted, value in values.items():
            section, _, key = dotted.partition(".")
            if section not in tables or (key not in tables[section] and dotted != "workers.count"):
                raise KeyError(f"no setting {dotted!r} to override")
            tables[section][key] = value
        return Settings(tables, self._profile_tables)


def shown(path):
    """A path for printing: relative to the project root when inside it."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve(path):
    """A settings path: relative ones are inside the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_profiles():
    profiles = {}
    for path in sorted(PROFILE_DIR.glob("*.toml")):
        with open(path, "rb") as f:
            profiles[path.stem] = tomllib.load(f)
    if not profiles:
        raise SystemExit(f"no driving-style profiles in {PROFILE_DIR}/")
    return profiles


def load(overrides=None):
    """Read and validate every config file; overrides: {"section.key": value}."""
    with open(SETTINGS_FILE, "rb") as f:
        tables = tomllib.load(f)
    cfg = Settings(tables, _read_profiles())
    return cfg.override(overrides) if overrides else cfg


def profile(name):
    """The merged rewards + eval bonuses of a profile, from the config files."""
    return load().profile(name)
