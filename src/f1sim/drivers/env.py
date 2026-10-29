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
- Stage B: a share of the sessions (`quali_share`) are short qualifying sessions instead of races: every car in its
  garage, out twice on new softs (f1env's default two runs), the flag after `quali_s` seconds.

Per car and step it gives what the replay needs: `valid` (the driver was in control before the step: the race had
started, the car was running and not parked in its garage; the pit lane counts, as the car is still in the race), `end`
(TERMINAL after a crash, a black flag or the finish; CUT when the race restarts under a car still going, which
bootstraps) and the reward amounts per term (`f1sim.drivers.rewards`), before the profile's weights. A qualifying run ends when the
car parks in its garage: that step's car isn't `valid` any more, so the replay doesn't bootstrap past it.

    state, obs = reset(bank, key, cfg)                         # obs (R, C, 209)
    state, obs, out, info = step(bank, state, action, key, cfg)  # action (R, C, 5) in [-1, 1]
    state, obs = from_race(bank, race_state, key, cfg)     # around a session f1env's F1Env started (evals)

Actions (as v2): steer; throttle and brake, each on above 0; DRS and ERS asked for above 0.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from f1env import observe
from f1env import pits as pt
from f1env.principal import quali_orders, quali_plan
from f1env.race import OUT, QUALIFYING, RACING, WRECK, RaceActions, RaceConfig, RaceState, pit_orders, race_step, start_race
from f1env.upgrades import random_levels
from f1env.weekend import start_qualifying

from ..replay import CONTINUE, CUT, TERMINAL
from . import principal, rewards

OBS = observe.size(observe.DRIVER_LAYOUT)  # 209
ACTIONS = 5
SLIPPING = 0.01  # wheelspin / lock-up beyond this (how far past the grip) counts as time slipping
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
    quali_share: float = 0.0  # sessions that are qualifying instead of a race (stage B: 0.2)
    quali: bool = False  # sessions may be qualifying (they are with quali_share > 0, or from F1Env's weekends)
    quali_s: float = 720.0  # a short session (Q3's 12 minutes)
    quali_runs: int = 2
    release_gap: float = 4.0  # s between cars leaving the garage
    quali_wet_share: float = 0.1
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
    plan: jax.Array  # (R, C, runs) qualifying: when each car leaves its garage (inf in races)


class Out(NamedTuple):
    amounts: jax.Array  # (R, C, terms)
    valid: jax.Array  # (R, C)
    end: jax.Array  # (R, C) int8
    lap_done: jax.Array  # (R, C) a lap completed (by a car the driver controlled)
    attack: jax.Array  # (R, C) under attack: a rival within 1 s behind
    lost: jax.Array  # (R, C) places lost on the track this step
    slipping: jax.Array  # (R, C) wheelspin or a lock-up


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


def _select(mask, new, old):
    """Per race: `new` where mask (R,), else `old`. A leaf shared by every race (the weather's random key) is taken
    from `new` if any race is."""

    def pick(n, o):
        n = jnp.broadcast_to(jnp.asarray(n, o.dtype), o.shape)
        if o.ndim == 0 or o.shape[0] != mask.shape[0]:
            return jnp.where(jnp.any(mask), n, o)
        return jnp.where(mask.reshape((-1,) + (1,) * (o.ndim - 1)), n, o)

    return jax.tree.map(pick, new, old)


def new_races(bank, key, cfg: EnvConfig, tid=None):
    """(RaceState, time limit (R,), tyre thresholds (R, C), qualifying plan (R, C, runs)) for R new sessions."""
    r, c = cfg.races, cfg.cars
    k_tid, k_laps, k_up, k_grid, k_race, k_worn, k_quali = jax.random.split(key, 7)
    tid = jax.random.randint(k_tid, (r,), 0, bank.length.shape[0]) if tid is None else jnp.asarray(tid, jnp.int32)
    laps = race_laps(bank, tid, k_laps, cfg)
    setup = random_levels(k_up, r, c, per_team=False) if cfg.upgrades else None
    if not cfg.random_grid:
        grid = jnp.broadcast_to(jnp.arange(c), (r, c))
    elif c == 1:
        grid = jax.random.randint(k_grid, (r, 1), 0, GRID_SLOTS)
    else:
        grid = jnp.argsort(jax.random.uniform(k_grid, (r, c)), axis=-1)
    race = _strong(principal.fixed_orders(start_race(bank, tid, laps, k_race, cfg.race, c, setup, grid=grid)))
    limit = (cfg.time_factor * bank.lap_time[tid] * laps + 60.0).astype(jnp.float32)
    plan = jnp.full((r, c, cfg.quali_runs), jnp.inf, jnp.float32)
    if cfg.quali_share > 0:
        k_start, k_plan, k_kind = jax.random.split(k_quali, 3)
        taking = jnp.ones((r, c), bool)
        quali = _strong(principal.fixed_orders(start_qualifying(bank, tid, k_start, taking, cfg.quali_s, cfg.race, setup, cfg.quali_wet_share)))
        is_q = jax.random.uniform(k_kind, (r,)) < cfg.quali_share
        race = _select(is_q, quali, race)
        limit = jnp.where(is_q, cfg.quali_s + 600.0, limit)
        q_plan = quali_plan(bank, tid, k_plan, taking, cfg.quali_s, cfg.quali_runs, cfg.release_gap)
        plan = jnp.where(is_q[:, None, None], q_plan, plan)
    return race, limit, principal.draw_worn(k_worn, (r, c), cfg.worn), plan


def _inputs(bank, race, readings, prev_edge, last, flags, cfg: EnvConfig):
    return observe.driver(bank, race, readings, prev_edge, last, (flags[..., 0], flags[..., 1], flags[..., 2]), cfg.race)


def _wrap(bank, race, limit, worn, plan, cfg: EnvConfig):
    r, c = race.s.shape
    z = jnp.zeros((r, c), jnp.float32)
    last = RaceActions(z, z, z, z, z)
    flags = jnp.zeros((r, c, 3), bool)
    readings = observe.lidar(bank, race)
    state = EnvState(race, readings.edge, last, flags, worn, limit, jnp.zeros(r, jnp.int32), rewards.new_memory(race.timing.place), plan)
    return state, _inputs(bank, race, readings, readings.edge, last, flags, cfg)


def reset(bank, key, cfg: EnvConfig, tid=None):
    race, limit, worn, plan = new_races(bank, key, cfg, tid)
    return _wrap(bank, race, limit, worn, plan, cfg)


def from_race(bank, race, key, cfg: EnvConfig, plan=None):
    """(EnvState, inputs) around a session started elsewhere: f1env's F1Env runs weekends (Q1-Q3 and the grid, 305 km
    races, points). plan: the session's qualifying plan (None in a race). The training principal and its fixed orders
    take over; a race is cut at `time_factor` x its lap-simulation time, a qualifying session 10 minutes after its flag."""
    r, c = race.s.shape
    race = _strong(principal.fixed_orders(race))
    is_q = race.session == QUALIFYING
    limit = jnp.where(is_q, race.session_end + 600.0, cfg.time_factor * bank.lap_time[race.tid] * race.target_laps + 60.0).astype(jnp.float32)
    plan = jnp.full((r, c, cfg.quali_runs), jnp.inf, jnp.float32) if plan is None else jnp.asarray(plan, jnp.float32)
    return _wrap(bank, race, limit, principal.draw_worn(key, (r, c), cfg.worn), plan, cfg)


def _restart(bank, s: EnvState, mask, key, cfg: EnvConfig) -> EnvState:
    """New sessions in place of those in `mask` (R,)."""
    race, limit, worn, plan = new_races(bank, key, cfg)
    z = jnp.zeros_like(s.last.steer)
    fresh = EnvState(race, s.prev_edge, RaceActions(z, z, z, z, z), jnp.zeros_like(s.flags), worn, limit, jnp.zeros_like(s.tick), rewards.new_memory(race.timing.place), plan)
    return _select(mask, fresh, s)


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
    decide = s.tick % PRINCIPAL_EVERY == 0
    is_q = race.session == QUALIFYING
    call = principal.calls(bank, race, s.worn_at, cfg.race)
    race = pit_orders(race, jnp.where((decide & ~is_q)[:, None], call, -1))
    if cfg.quali or cfg.quali_share > 0:  # qualifying: f1env's default runs (out-lap, flying lap, in-lap, twice)
        runs = quali_orders(bank, race, s.plan, cfg.race)
        race = race._replace(pits=_select(decide & is_q, runs.pits, race.pits))
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
    nxt = EnvState(new, s.prev_edge, acts, flags, worn_at, s.limit, s.tick + 1, mem, s.plan)

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
    dyn = ev.dynamics
    slipping = (dyn.wheelspin > SLIPPING) | (dyn.lockup_front > SLIPPING) | (dyn.lockup_rear > SLIPPING)
    out = Out(amounts, valid, end, valid & ev.lap_done, valid & ev.records.under_attack, jnp.where(valid, ev.records.lost, 0), valid & slipping)
    return nxt, obs, out, info


@lru_cache(maxsize=32)
def compiled(name: str, cfg: EnvConfig):
    """reset / from_race / race_done for one config, jit-compiled once and reused (evals call them every session)."""
    return jax.jit(partial({"reset": reset, "from_race": from_race, "race_done": race_done}[name], cfg=cfg))
