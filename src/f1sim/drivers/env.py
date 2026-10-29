"""The drivers' training environment (Phase 2, section 3): f1env races, batched and jit-compiled.

- R races x C cars: C = 1 is stage A (alone), C = 22 stage B (the full grid). Each race is on a random track of the
  bank, from 5 laps up to the full 305 km, with at least `pit_share` of races long enough to need a pit stop; a random
  grid slot and a standing start; f1env's weather (rain in 20% of races).
- Every car gets random upgrade levels each race: each category 0-10, drawn per car.
- The scripted training principal calls the pit stops (`f1sim.drivers.principal`); pace normal, no team orders, ERS
  "deploy".
- Crashed cars rejoin from the pit exit after ~5 s (f1env's training respawn) and start a new episode.
- A race that's over, or past its time limit, restarts at once on a new track inside the same jitted step, so no batch
  slot waits.

Per car and step it gives what the replay needs: `valid` (the driver was in control before the step: the race had
started, the car was running and not parked in its garage; the pit lane counts, as the car is still in the race), `end`
(TERMINAL after a crash, a black flag or the finish; CUT when the race restarts under a car still going, which
bootstraps) and the reward amounts per term (`f1sim.drivers.rewards`), before the profile's weights.

    state, obs = reset(bank, key, cfg)                         # obs (R, C, 209)
    state, obs, out, info = step(bank, state, action, key, cfg)  # action (R, C, 5) in [-1, 1]

Actions (as v2): steer; throttle and brake, each on above 0; DRS and ERS asked for above 0.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from f1env import observe
from f1env import pits as pt
from f1env.race import OUT, RACING, WRECK, RaceActions, RaceConfig, RaceState, pit_orders, race_step, start_race
from f1env.upgrades import random_levels

from ..replay import CONTINUE, CUT, TERMINAL
from . import principal, rewards

OBS = observe.size(observe.DRIVER_LAYOUT)  # 209
ACTIONS = 5
LIDAR_BEAMS = observe.DRIVER_LAYOUT["lidar_edge"]
GRID_SLOTS = 22
PRINCIPAL_EVERY = 20  # steps: the principal decides once a second


class EnvConfig(NamedTuple):
    races: int = 8
    cars: int = 1
    min_laps: int = 5
    race_km: float = 305.0  # the longest race
    pit_share: float = 0.5  # races long enough to need a pit stop on their first set
    pit_km: float = 90.0  # ... which is more than this (mediums last ~100 km, boxed at ~10%)
    fixed_laps: int = 0  # evals: every race this long (0: drawn)
    time_factor: float = 2.5  # a race is cut (and bootstrapped) at this x its lap-simulation time, plus a minute
    worn: tuple = (0.08, 0.12)  # the principal boxes each stint at a tyre life drawn from this range
    upgrades: bool = True  # random upgrade levels per car (False: the default car, for evals)
    random_grid: bool = True  # False: car k in slot k (evals)
    auto_reset: bool = True  # False: races that are over stay over (evals)
    race: RaceConfig = RaceConfig(respawn=True)


class EnvState(NamedTuple):
    race: RaceState
    prev_edge: jax.Array  # (R, C, beams) the lidar's edge readings for the last inputs (closing speeds)
    last: RaceActions  # (R, C) the actions sent last step
    flags: jax.Array  # (R, C, 3) DRS available, DRS open, ERS deploying, from the last step
    worn_at: jax.Array  # (R, C) the tyre life this stint is boxed at
    limit: jax.Array  # (R,) the clock at which the race is cut
    tick: jax.Array  # (R,) steps since the race started
    mem: rewards.Memory


class Out(NamedTuple):
    amounts: jax.Array  # (R, C, terms)
    valid: jax.Array  # (R, C)
    end: jax.Array  # (R, C) int8


def to_race_actions(a) -> RaceActions:
    """(..., 5) in [-1, 1] -> f1env's RaceActions."""
    return RaceActions(a[..., 0], jnp.maximum(a[..., 1], 0.0), jnp.maximum(a[..., 2], 0.0), a[..., 3], a[..., 4])


def race_laps(bank, tid, key, cfg: EnvConfig):
    """(R,) laps: a short race (up to the pit distance) or a long one (beyond it, up to 305 km)."""
    length = bank.length[tid]
    if cfg.fixed_laps:
        return jnp.full(tid.shape, cfg.fixed_laps, jnp.int32)
    full = jnp.ceil(cfg.race_km * 1000.0 / length).astype(jnp.int32)
    need = jnp.ceil(cfg.pit_km * 1000.0 / length).astype(jnp.int32)
    k_long, k_laps = jax.random.split(key)
    long = jax.random.uniform(k_long, tid.shape) < cfg.pit_share
    lo = jnp.where(long, jnp.clip(need + 1, cfg.min_laps, full), cfg.min_laps)
    hi = jnp.where(long, full, jnp.clip(need, cfg.min_laps, full))
    laps = lo + jnp.floor(jax.random.uniform(k_laps, tid.shape) * (hi - lo + 1)).astype(jnp.int32)
    return jnp.minimum(laps, hi)


def _strong(tree):
    """The same arrays without weak types, so a state keeps its types through scans and conds."""
    return jax.tree.map(lambda a: jnp.asarray(a).astype(jnp.asarray(a).dtype), tree)


def new_races(bank, key, cfg: EnvConfig, tid=None):
    """(RaceState, time limit (R,), tyre thresholds (R, C)) for R new races."""
    r, c = cfg.races, cfg.cars
    k_tid, k_laps, k_up, k_grid, k_race, k_worn = jax.random.split(key, 6)
    tid = jax.random.randint(k_tid, (r,), 0, bank.length.shape[0]) if tid is None else jnp.asarray(tid, jnp.int32)
    laps = race_laps(bank, tid, k_laps, cfg)
    setup = random_levels(k_up, r, c, per_team=False) if cfg.upgrades else None
    if not cfg.random_grid:
        grid = jnp.broadcast_to(jnp.arange(c), (r, c))
    elif c == 1:
        grid = jax.random.randint(k_grid, (r, 1), 0, GRID_SLOTS)
    else:
        grid = jnp.argsort(jax.random.uniform(k_grid, (r, c)), axis=-1)
    race = principal.fixed_orders(start_race(bank, tid, laps, k_race, cfg.race, c, setup, grid=grid))
    limit = cfg.time_factor * bank.lap_time[tid] * laps + 60.0
    return _strong(race), limit.astype(jnp.float32), principal.draw_worn(k_worn, (r, c), cfg.worn)


def _inputs(bank, race, readings, prev_edge, last, flags, cfg: EnvConfig):
    return observe.driver(bank, race, readings, prev_edge, last, (flags[..., 0], flags[..., 1], flags[..., 2]), cfg.race)


def reset(bank, key, cfg: EnvConfig, tid=None):
    race, limit, worn = new_races(bank, key, cfg, tid)
    z = jnp.zeros((cfg.races, cfg.cars), jnp.float32)
    last = RaceActions(z, z, z, z, z)
    flags = jnp.zeros((cfg.races, cfg.cars, 3), bool)
    readings = observe.lidar(bank, race)
    state = EnvState(race, readings.edge, last, flags, worn, limit, jnp.zeros(cfg.races, jnp.int32), rewards.new_memory(race.timing.place))
    return state, _inputs(bank, race, readings, readings.edge, last, flags, cfg)


def _restart(bank, s: EnvState, mask, key, cfg: EnvConfig) -> EnvState:
    """New races in place of those in `mask` (R,)."""
    race, limit, worn = new_races(bank, key, cfg)

    def pick(new, old):
        new = jnp.broadcast_to(jnp.asarray(new, old.dtype), old.shape)
        if old.ndim == 0 or old.shape[0] != mask.shape[0]:  # one for every race (the weather's random key)
            return jnp.where(jnp.any(mask), new, old)
        return jnp.where(mask.reshape((-1,) + (1,) * (old.ndim - 1)), new, old)

    race = jax.tree.map(pick, race, s.race)
    return EnvState(
        race,
        s.prev_edge,
        jax.tree.map(lambda a: pick(0.0, a), s.last),
        pick(False, s.flags),
        pick(worn, s.worn_at),
        pick(limit, s.limit),
        pick(0, s.tick),
        jax.tree.map(pick, rewards.new_memory(race.timing.place), s.mem),
    )


def controlled(race):
    """(R, C) the driver drives (or the autopilot, in the pit lane, for a car still in the race)."""
    return (race.status == RACING) & (race.clock >= 0)[:, None] & ~pt.parked(race.pits)


def race_done(bank, race, limit, cfg: EnvConfig):
    """(R,) nothing left to race: no car running or about to rejoin, past the time limit, or the flag is out and only
    cars that already crashed (and rejoined) are still going. A rejoined car can't finish, so it stops counting once it
    has covered the race distance."""
    running = race.status == RACING
    rejoining = cfg.race.respawn & race.dnf & ~race.flags.black & ((race.status == WRECK) | (race.status == OUT))
    laps = jnp.floor(race.line / bank.length[race.tid][:, None])
    through = running & race.dnf & (laps >= race.target_laps[:, None])
    going = (running & ~through) | rejoining
    return ~jnp.any(going, axis=-1) | (race.clock >= limit) | (race.flag & ~jnp.any(running & ~race.dnf, axis=-1))


def step(bank, s: EnvState, action, key, cfg: EnvConfig):
    k_worn, k_reset = jax.random.split(key)
    acts = to_race_actions(action)
    race = s.race
    call = principal.calls(bank, race, s.worn_at, cfg.race)
    race = pit_orders(race, jnp.where((s.tick % PRINCIPAL_EVERY == 0)[:, None], call, -1))
    live = ~race_done(bank, race, s.limit, cfg)  # without auto-reset (evals) a race that's over stays over
    valid = controlled(race) & live[:, None]

    new, ev = race_step(bank, race, acts, cfg.race)
    new = jax.tree.map(lambda a, b: a.astype(b.dtype), new, race)
    amounts, mem = rewards.amounts(ev, acts, s.last, s.mem, valid, ev.respawned)
    terminal = valid & (ev.crashed | ev.finished | ev.black_flag)
    over = race_done(bank, new, s.limit, cfg)
    end = jnp.where(terminal, TERMINAL, jnp.where(valid & over[:, None], CUT, CONTINUE)).astype(jnp.int8)
    worn_at = jnp.where(ev.pit_fitted, principal.draw_worn(k_worn, s.worn_at.shape, cfg.worn), s.worn_at)
    flags = jnp.stack([ev.drs_available, ev.drs_open, ev.ers_on], axis=-1)
    nxt = EnvState(new, s.prev_edge, acts, flags, worn_at, s.limit, s.tick + 1, mem)

    restarted = jnp.zeros_like(over)
    if cfg.auto_reset:
        nxt = jax.lax.cond(jnp.any(over), lambda: _restart(bank, nxt, over, k_reset, cfg), lambda: nxt)
        restarted = over
    readings = observe.lidar(bank, nxt.race)
    prev = jnp.where(restarted[:, None, None], readings.edge, nxt.prev_edge)
    obs = _inputs(bank, nxt.race, readings, prev, nxt.last, nxt.flags, cfg)
    nxt = nxt._replace(prev_edge=readings.edge)

    ref = bank.lap_time[s.race.tid][:, None]
    info = {
        "amounts": jnp.sum(amounts, axis=(0, 1)),
        "valid": jnp.sum(valid),
        "crashes": jnp.sum(valid & ev.crashed),
        "finishes": jnp.sum(valid & ev.finished),
        "black_flags": jnp.sum(valid & ev.black_flag),
        "laps": jnp.sum(valid & ev.lap_done),
        "laps_counted": jnp.sum(valid & ev.lap_counted),
        "lap_ratio": jnp.sum(jnp.where(valid & ev.lap_counted, ev.lap_time / ref, 0.0)),
        "pit_stops": jnp.sum(valid & ev.pit_in),
        "dead_tyre": jnp.sum(valid & ev.tyre_dead),
        "races": jnp.sum(over & live),
    }
    info = {k: v.astype(jnp.float32) for k, v in info.items()}
    return nxt, obs, Out(amounts, valid, end), info
