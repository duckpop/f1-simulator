"""The drivers' training environment (Phase 2, section 3): f1env races, batched and jit-compiled.

- R races x C cars: C = 1 is stage A (alone), C = 22 stage B (the full grid). Each race is on a random track of the
  bank, from 5 laps up to the full 305 km, with at least `pit_share` of races long enough to need a pit stop; a random
  grid slot and a standing start; f1env's weather (rain in 20% of races).
- Every car gets random upgrade levels each race: each category 0-10, drawn per car.
- The scripted training principal calls the pit stops (`f1sim.drivers.principal`); pace normal, no team orders, ERS
  "deploy".
- A crashed car is out for the rest of its race (`respawn` false, since 2026-11-01: a crash isn't just a short setback);
  with respawn on it rejoins from the pit exit after ~5 s (f1env's training respawn) as a new episode.
- A race that's over, or past its time limit, restarts at once on a new track inside the same jitted step, so no batch
  slot waits.
- Stage B: a share of the sessions (`quali_share`) are short qualifying sessions instead of races: every car in its
  garage, out twice on new softs (f1env's default two runs), the flag after `quali_s` seconds.
- Field sizes (`field_sizes`, stage B since 2026-11-06): each session draws how many cars take part (e.g. 6, 12, 18 or
  22); the others are out (f1env's OUT: not on the track, not seen, not stored). The learning seats keep their share of
  the field (with `learners_first`: all of them, frozen cars filling the rest), the `keep` cars (stage B: the seats the replay stores) first, and the cars in it fill the front slots.
- Spread starts (`spread_share` of the races): the cars start spread around the lap at speed (`spread_speed` x the
  reference pace there) in a random order, lights already out, instead of standing on the grid: racing in traffic
  without a lap-1 pile-up (v2 had the same, its "random" spawn). Each car's first, partial lap isn't timed.

Per car and step it gives what the replay needs: `valid` (the driver was in control before the step: the race had
started, the car was running and not parked in its garage; the pit lane counts, as the car is still in the race), `end`
(TERMINAL after a crash, a black flag or the finish; CUT when the race restarts under a car still going, which
bootstraps) and the reward amounts per term (`f1sim.drivers.rewards`), before the profile's weights. A qualifying run ends when the
car parks in its garage: that step's car isn't `valid` any more, so the replay doesn't bootstrap past it.

    state, obs = reset(bank, key, cfg)                         # obs (R, C, 188)
    state, obs, out, info = step(bank, state, action, key, cfg)  # action (R, C, 5) in [-1, 1]
    state, obs = from_race(bank, race_state, key, cfg)     # around a session f1env's F1Env started (evals)

Actions (as v2): steer; throttle and brake, each on above 0; DRS and ERS asked for above 0.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from f1env import observe
from f1env import orders as od
from f1env import pits as pt
from f1env.principal import quali_orders, quali_plan
from f1env import racecontrol as rc
from f1env import tyres as ty
from f1env.race import OUT, QUALIFYING, RACE, RACING, WRECK, RaceActions, RaceConfig, RaceState, _start_timing, pit_orders, race_step, start_race
from f1env.sensors import MARCH_STEPS, beam_angles, locate_global, observed_beams, sense
from f1env.upgrades import random_levels
from f1env.weekend import start_qualifying

from ..replay import CONTINUE, CUT, TERMINAL
from . import closecalls, principal, rewards

OBS = observe.size(observe.DRIVER_LAYOUT)  # 188 (186 until 2026-11-16, before the stuck timer's 2; 184 until 2026-11-12, before the blend line's 2; 209 until 2026-11-05, then 234 with the cars around and the safety car
# appended, until 2026-11-06: the lidar's closing speeds removed, its distances on a log scale)


def _storage_spec():
    """Per input, how the replay stores it (`replay.Codec`): (lo, hi, square-root scale) in 8 bits, or None for float16.
    Ranges are the inputs' own (from how f1env builds them, with room), not fitted to data, so they hold for any run.
    The car's state and the track preview stay float16: in 8 bits they moved a trained driver's steering by up to 0.06
    to 0.12 (99th percentile); every other group moved it by at most 0.016, about a sixth of stage B's action noise
    (measured 2026-11-06 on the linear lidar). The lidar is already on f1env's log scale, so 8 bits linear over it keep
    ~1.7% of (5 m + the distance): ~0.12 m at 2 m. The other distances (gaps) use the square-root scale."""
    unit, sign, dist = (0.0, 1.0, False), (-1.0, 1.0, False), (0.0, 1.0, True)
    spec = {
        "lidar_edge": [unit] * 50,  # log scale (f1env's lidar_scale)
        "lidar_barrier": [unit] * 50,
        "car": [None] * 7,
        "race": [unit, (0.0, 4.2, False)],  # place; cars within 50 m / 5 (21 at most)
        "drs": [unit] * 3,
        "ers": [unit] * 2,
        "mirrors": [(-1.1, 1.1, False)] * 3,
        "preview": [None] * 7,
        "alongside": [dist, sign, dist, sign],
        "upgrades": [(-0.5, 1.5, False)] * 4,
        "tyres": [unit] * 4 + [(0.0, 2.0, False), unit],  # compound one-hot, laps on the set / 40, life
        "progress": [(-0.25, 1.25, False), unit],
        "orders": [unit] * 11,
        "pace_delta": [sign] * 2,
        "status": [unit, unit, unit, (0.0, 1.2, False), unit, unit],
        "cars_ahead": [dist, (-4.0, 4.0, False), (-4.0, 4.0, False), unit, unit] * observe.AHEAD,
        "cars_behind": [dist, (-4.0, 4.0, False), (-4.0, 4.0, False)] * observe.BEHIND,
        "safety_car": [dist, unit, unit, unit],
        "blend": [unit, sign],  # on the blend line's stretch; m beyond it / 5
        "stuck": [unit, unit],  # the stuck timer / its time-out; the wait level
    }
    if list(spec) != list(observe.DRIVER_LAYOUT) or any(len(spec[k]) != n for k, n in observe.DRIVER_LAYOUT.items()):
        raise ValueError("f1env's driver inputs changed: update drivers.env._storage_spec")
    return [x for k in observe.DRIVER_LAYOUT for x in spec[k]]


@lru_cache(maxsize=1)
def storage_codec():
    """The replay's packing of the driver inputs (8 bits for 174 of the 188)."""
    from ..replay import make_codec

    return make_codec(_storage_spec())
ACTIONS = 5
SLIPPING = 0.01  # wheelspin / lock-up beyond this (how far past the grip) counts as time slipping
LIDAR_BEAMS = observe.DRIVER_LAYOUT["lidar_edge"]
LIDAR_CHANNELS = 2  # the lidar's readings per beam, first in the inputs channel by channel: edge, barrier
GRID_SLOTS = 22
PRINCIPAL_EVERY = 20  # steps: the principal decides once a second


class EnvConfig(NamedTuple):
    races: int = 8
    cars: int = 1
    min_laps: int = 5
    race_km: float = 305.0  # the longest race
    pit_share: float = 0.5  # races long enough to need a pit stop on their first set
    pit_km: float = 115.0  # ... which is more than this (a medium does ~127 km, boxed at ~10%)
    length_skew: float = 1.0  # laps drawn as lo + u**skew x (hi - lo), u uniform: 1 even, above 1 more short races
    two_compound_km: float = 0.0  # races shorter than this skip the two-compound stop (the principal's rule)
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
    learners: tuple = ()  # (C,) the cars that keep a race going ((): every car); stage B: the learning seats, so a
    # race whose learners are all out restarts at once instead of waiting for its frozen cars
    field_sizes: tuple = ()  # cars per session, one drawn per session ((): every car)
    learners_first: bool = False  # True: every learning seat takes part in every session, frozen cars fill the rest
    keep: tuple = ()  # (C,) cars that take part whenever their group (learning / frozen) has room, before the others
    spread_share: float = 0.0  # races that start spread around the lap at speed instead of on the grid
    spread_speed: tuple = (0.6, 0.9)  # x the reference pace where a spread car starts
    spread_lateral: float = 1.0  # m, at most, off the centreline
    sensor_noise: float = 0.0  # training: each sensed distance x (1 + U(-n, n)), every step (`noisy`); evals: 0
    start_tyres: tuple = ()  # (soft, medium, hard) shares of the dry compound each car starts a race on, drawn per car
    # (training, 2026-11-23); (): every car on mediums (f1env's start, the evals). Wet starts stay on wets; qualifying on softs
    random_orders: bool = False  # stage C: a random principal gives the learners team orders in races (`random_orders`), and the order terms pay
    order_every: tuple = (30.0, 90.0)  # s between one head's orders (pace, team, ERS), drawn per order
    order_rest: float = 0.5  # the share of those draws that are "no order" (pace normal, no team order, ERS deploy)
    march_steps: int = MARCH_STEPS  # the lidar's sphere-tracing steps per beam (f1env: 128). 64: the simulation ~35% faster, 0.07% of beams read wrong (2026-11-24)
    crash_left_km: float = 0.0  # a crash or black flag costs this much more of its cost per km of the race it ends early (`rewards.race_left`)


def _lidar(bank, race, cfg: EnvConfig):
    """f1env's `observe.lidar` (the safety car one more body), its lidar marching `cfg.march_steps` steps per beam."""
    if cfg.march_steps == MARCH_STEPS:
        return observe.lidar(bank, race)
    x, y, yaw, v, active = rc.bodies_with_safety_car(bank, race)
    s = jnp.concatenate([race.s, race.sc.s[:, None]], axis=1)
    r = sense(bank, race.tid, x, y, yaw, v, s, active, jnp.asarray(beam_angles()[observed_beams()]), steps=cfg.march_steps)
    return jax.tree.map(lambda a: a[:, :-1], r)


def stage_race(race: RaceConfig, sb: dict) -> RaceConfig:
    """A stage's race rules over `race`, the same in its training races, its evals and `watch`: from its settings
    (`[stage_b]`, the later stages' over it) the stuck rule's time stuck_s (f1env: 6 s) and its shorter one for a car
    stopped on the track stuck_track_s, and sc_band_kmh, the safety car queue's limit over its speed (f1env's
    queue_margin; unset or 0: f1env's own, 15 km/h)."""
    if "stuck_s" in sb:
        race = race._replace(stuck_s=float(sb["stuck_s"]))
    if "stuck_track_s" in sb:
        race = race._replace(stuck_track_s=float(sb["stuck_track_s"]))
    if sb.get("sc_band_kmh"):
        race = race._replace(control=race.control._replace(queue_margin=float(sb["sc_band_kmh"]) / 3.6))
    return race


class EnvState(NamedTuple):
    race: RaceState
    last: RaceActions  # (R, C) the actions sent last step
    wall_gap: jax.Array  # (R, C) m from each body to its nearest barrier after the last step (barrier close calls)
    flags: jax.Array  # (R, C, 3) DRS available, DRS open, ERS deploying, from the last step
    worn_at: jax.Array  # (R, C) the tyre life this stint is boxed at
    limit: jax.Array  # (R,) the clock at which the race is cut
    tick: jax.Array  # (R,) steps since the race started
    mem: rewards.Memory
    plan: jax.Array  # (R, C, runs) qualifying: when each car leaves its garage (inf in races)
    pick: jax.Array  # (2,) new sessions draw their track from bank slots [pick[0], pick[0] + pick[1]) (track rotation)
    order_due: jax.Array  # (R, C, 3) the race clock at which the random principal next sets each head (pace, team, ERS); stage C


class Out(NamedTuple):
    amounts: jax.Array  # (R, C, terms)
    valid: jax.Array  # (R, C)
    end: jax.Array  # (R, C) int8
    lap_done: jax.Array  # (R, C) a lap completed (by a car the driver controlled)
    attack: jax.Array  # (R, C) under attack: a rival within 1 s behind
    lost: jax.Array  # (R, C) places lost on the track this step
    slipping: jax.Array  # (R, C) wheelspin or a lock-up
    metres: jax.Array  # (R, C) progress along the centreline on the track (the progress term pays it against the pace)
    wear: jax.Array  # (R, C) % of tyre life used (the tyre_wear term charges only what's beyond the rated wear)
    crashed: jax.Array  # (R, C) a crash
    hit: jax.Array  # (R, C) a contact crash the car was mostly not to blame for (f1env's blame under a half)
    warnings: jax.Array  # (R, C, 7) warnings given this step by kind (`WARNING_KINDS`)
    black_by: jax.Array  # (R, C, 7) a black flag this step, by the kind of the warning that gave it (the 4th)
    # clean running (since 2026-11-21, the eval's race pace): racing on the track with a clear road (no car within
    # CLEAN_GAP_S ahead), no safety car, blue flag or standing start; its seconds, and the lap sim's for its metres
    clean_s: jax.Array = 0.0  # (R, C)
    clean_sim_s: jax.Array = 0.0  # (R, C)
    # attacking (since 2026-11-21): racing within CLEAN_GAP_S of the car ahead with no safety car; places gained on the
    # track this step (a pass)
    attacking: jax.Array = False  # (R, C)
    gained: jax.Array = 0  # (R, C)


CLEAN_GAP_S = 1.0  # a car this close ahead (s at the car's speed) is traffic: its pace isn't the driver's own
STEP_S = 0.05
EXACT_END = 100  # evals: steps per block between checks that every race is done (a session ends within 100 steps of it, not 2,000)

WARNING_KINDS = ("speeding", "passing", "blue", "slow", "blend", "close", "far")  # f1env's warning bits 1, 2, 4, 8, 16, 32, 64 in order
# (blend: 2026-11-12; close, far: the safety car queue, 2026-11-19; slow: not closing on the safety car since then)


def clean_running(ev, valid, pace, launching, blue):
    """(R, C) seconds of clean running this step, the lap sim's seconds for the same metres, and attacking: racing
    on the road with traffic close ahead (see `Out`)."""
    speed = jnp.maximum(ev.progress, 0.0) / STEP_S
    clear = ev.gap_ahead > CLEAN_GAP_S * speed
    road = valid & ~ev.off_track & ~ev.in_pit & (ev.sc_phase == rc.NONE)[:, None] & (ev.progress > 0)
    clean = road & clear & ~launching & ~blue
    return jnp.where(clean, STEP_S, 0.0), jnp.where(clean, ev.progress / pace, 0.0), road & ~clear


def warning_kinds(warned):
    """(...) warning bits -> (..., 7) float: one per kind given (`WARNING_KINDS`)."""
    warned = jnp.asarray(warned, jnp.int32)
    return jnp.stack([(warned >> k) & 1 for k in range(len(WARNING_KINDS))], axis=-1).astype(jnp.float32)


def to_race_actions(a) -> RaceActions:
    """(..., 5) in [-1, 1] -> f1env's RaceActions."""
    return RaceActions(a[..., 0], jnp.maximum(a[..., 1], 0.0), jnp.maximum(a[..., 2], 0.0), a[..., 3], a[..., 4])


def random_start_tyres(race: RaceState, key, shares) -> RaceState:
    """Each car on a dry start (mediums) on soft, medium or hard, drawn per car with these shares; wets stay."""
    t = race.tyres
    draw = jax.random.choice(key, 3, t.compound.shape, p=jnp.asarray(shares, jnp.float32) / sum(shares)).astype(t.compound.dtype)
    dry = t.compound == ty.MEDIUM
    compound = jnp.where(dry, draw, t.compound)
    used = jnp.where(dry, jnp.left_shift(1, compound).astype(t.used.dtype), t.used)
    return race._replace(tyres=t._replace(compound=compound, used=used))


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
    u = jax.random.uniform(k_laps, tid.shape) ** cfg.length_skew  # skew > 1: towards the short end of each range
    laps = lo + jnp.floor(u * (hi - lo + 1)).astype(jnp.int32)
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


def all_tracks(bank):
    return jnp.array([0, bank.length.shape[0]], jnp.int32)


def new_races(bank, key, cfg: EnvConfig, tid=None, pick=None):
    """(RaceState, time limit (R,), tyre thresholds (R, C), qualifying plan (R, C, runs)) for R new sessions, each on a
    random track from bank slots pick = (first, count) (default: all)."""
    r, c = cfg.races, cfg.cars
    k_tid, k_laps, k_up, k_grid, k_race, k_worn, k_quali = jax.random.split(key, 7)
    if tid is None:
        pick = all_tracks(bank) if pick is None else pick
        tid = pick[0] + jax.random.randint(k_tid, (r,), 0, pick[1])
    tid = jnp.asarray(tid, jnp.int32)
    laps = race_laps(bank, tid, k_laps, cfg)
    setup = random_levels(k_up, r, c, per_team=False) if cfg.upgrades else None
    k_field, k_spread, k_grid = jax.random.split(k_grid, 3)
    present = field(k_field, cfg)
    if not cfg.random_grid:
        grid = jnp.broadcast_to(jnp.arange(c), (r, c))
    elif c == 1:
        grid = jax.random.randint(k_grid, (r, 1), 0, GRID_SLOTS)
    else:  # a random order, the cars taking part in the front slots
        grid = _ranks(jax.random.uniform(k_grid, (r, c)) + 2.0 * ~present)
    race = start_race(bank, tid, laps, k_race, cfg.race, c, setup, grid=grid)
    if cfg.start_tyres:
        race = random_start_tyres(race, jax.random.fold_in(k_race, 17), cfg.start_tyres)
    race = race._replace(status=jnp.where(present, race.status, OUT))
    if cfg.spread_share > 0:
        k_which, k_place = jax.random.split(k_spread)
        spread = jax.random.uniform(k_which, (r,)) < cfg.spread_share
        race = _select(spread, spread_start(bank, race, present, k_place, cfg), race)
    race = _strong(principal.fixed_orders(race))
    limit = (cfg.time_factor * bank.lap_time[tid] * laps + 60.0).astype(jnp.float32)
    plan = jnp.full((r, c, cfg.quali_runs), jnp.inf, jnp.float32)
    if cfg.quali_share > 0:
        k_start, k_plan, k_kind = jax.random.split(k_quali, 3)
        taking = present
        quali = _strong(principal.fixed_orders(start_qualifying(bank, tid, k_start, taking, cfg.quali_s, cfg.race, setup, cfg.quali_wet_share)))
        is_q = jax.random.uniform(k_kind, (r,)) < cfg.quali_share
        race = _select(is_q, quali, race)
        limit = jnp.where(is_q, cfg.quali_s + 600.0, limit)
        q_plan = quali_plan(bank, tid, k_plan, taking, cfg.quali_s, cfg.quali_runs, cfg.release_gap)
        plan = jnp.where(is_q[:, None, None], q_plan, plan)
    return race, limit, principal.draw_worn(k_worn, (r, c), cfg.worn), plan


def _ranks(x):
    """(R, C) each car's rank by x along the last axis (0: the smallest)."""
    return jnp.argsort(jnp.argsort(x, axis=-1), axis=-1)


def field(key, cfg: EnvConfig):
    """(R, C) the cars taking part in each new session: all of them, or with `field_sizes` one size drawn per session.
    The learning seats (`learners`) keep their share of the field (at least one), or with `learners_first` all take
    part (as many as fit), the frozen ones fill the rest, and within each group the `keep` cars go first, then a random
    pick."""
    r, c = cfg.races, cfg.cars
    if not cfg.field_sizes or c == 1:
        return jnp.ones((r, c), bool)
    k_size, k_pick = jax.random.split(key)
    sizes = jnp.asarray(cfg.field_sizes, jnp.int32)
    n = jnp.clip(sizes[jax.random.randint(k_size, (r,), 0, sizes.shape[0])], 1, c)
    learning = jnp.asarray(cfg.learners if cfg.learners else (True,) * c, bool)
    keep = jnp.asarray(cfg.keep if cfg.keep else (False,) * c, bool)
    n_l, n_f = jnp.sum(learning), c - jnp.sum(learning)
    want_l = jnp.minimum(n, n_l) if cfg.learners_first else jnp.clip(jnp.round(n * n_l / c).astype(jnp.int32), jnp.minimum(1, n_l), n_l)
    take_f = jnp.clip(n - want_l, 0, n_f)
    take_l = jnp.minimum(n - take_f, n_l)
    u = jax.random.uniform(k_pick, (r, c)) - 2.0 * keep
    rank_l = _ranks(jnp.where(learning, u, jnp.inf))
    rank_f = _ranks(jnp.where(learning, jnp.inf, u))
    return jnp.where(learning, rank_l < take_l[:, None], rank_f < take_f[:, None])


SPREAD_SINCE_S = 10.0  # a spread start's clock: the lights went out this long ago (past the launch window)


def spread_start(bank, race: RaceState, present, key, cfg: EnvConfig) -> RaceState:
    """The races' cars taking part spread around the lap instead of on the grid: evenly spaced (each moved by up to a
    third of the spacing) in a random order from a random point, up to `spread_lateral` m off the centreline, heading
    along it at `spread_speed` x the reference pace there, the lights out. Every car is short of the line (as on the
    grid), so the first crossing starts lap 1; that partial lap isn't timed."""
    r, c = present.shape
    k_order, k_jit, k_off, k_lat, k_v, k_ctl = jax.random.split(key, 6)
    tid = race.tid
    t = tid[:, None]
    length = bank.length[tid][:, None]
    n = jnp.maximum(jnp.sum(present, axis=-1, keepdims=True), 1)
    order = _ranks(jax.random.uniform(k_order, (r, c)) + 2.0 * ~present)
    gap = length / n
    s = (order + jax.random.uniform(k_jit, (r, c), minval=-0.33, maxval=0.33)) * gap + jax.random.uniform(k_off, (r, 1)) * length
    s = jnp.mod(s, length)
    idx = jnp.clip(jnp.round(s / bank.ds[t]).astype(jnp.int32), 0, bank.line_n[t] - 1)
    heading = bank.line_heading[t, idx]
    lat = jax.random.uniform(k_lat, (r, c), minval=-1.0, maxval=1.0) * cfg.spread_lateral
    x = bank.line_x[t, idx] - jnp.sin(heading) * lat
    y = bank.line_y[t, idx] + jnp.cos(heading) * lat
    v = bank.lap_speed[t, idx] * jax.random.uniform(k_v, (r, c), minval=cfg.spread_speed[0], maxval=cfg.spread_speed[1])
    on = present
    old = race.cars
    cars = old._replace(x=jnp.where(on, x, old.x), y=jnp.where(on, y, old.y), yaw=jnp.where(on, heading, old.yaw), vx=jnp.where(on, v, 0.0))
    pos = locate_global(bank, jnp.broadcast_to(t, cars.x.shape), cars.x, cars.y)
    line = jnp.where(on, pos.s - length, race.line)  # short of the line: lap 1 starts at the first crossing
    sc, flags = rc.new_control(k_ctl, line, cfg.race.control)
    zero = jnp.zeros_like(race.clock)
    since = zero + SPREAD_SINCE_S  # the lights went out a while ago: no launch to pay
    return race._replace(
        cars=cars, idx=pos.idx, s=pos.s, line=line, best_line=line, clock=since, lights_start=zero - cfg.race.lights_s - 1.0, lights_out=zero,
        lap_valid=jnp.zeros_like(race.lap_valid), sc=sc, flags=flags, timing=_start_timing(bank, tid, pos.s, line, c),
    )


def noisy(readings, key, n: float):
    """Sensor noise on the lidar: its edge and barrier distances (in metres, before f1env's log scale) x (1 + U(-n, n)),
    drawn per beam per step. The other cars and the safety car get theirs on the inputs (`_noise_seen`)."""
    k_edge, k_bar = jax.random.split(key)
    u = lambda k, shape: 1.0 + jax.random.uniform(k, shape, minval=-n, maxval=n)
    return readings._replace(edge=readings.edge * u(k_edge, readings.edge.shape), barrier=readings.barrier * u(k_bar, readings.barrier.shape))


# the inputs about another car or the safety car that get sensor noise: per group of f1env's driver inputs, (inputs per
# car, the ones that get noise: gap, sideways, relative speed; not the brake or wreck flags). The first of each car's
# inputs is its distance, 1 when nobody's there: then all of that car's stay exact, so the noise never makes up a car
_SEEN = {"cars_ahead": (5, (0, 1, 2)), "cars_behind": (3, (0, 1, 2)), "mirrors": (3, (0, 1, 2)), "alongside": (2, (0, 1)), "safety_car": (4, (0, 1))}


def _seen_mask():
    """(noisy, gate): per driver input whether it gets noise, and the input saying whether its car is there."""
    if set(_SEEN) - set(observe.DRIVER_LAYOUT) or any(observe.DRIVER_LAYOUT[k] % per for k, (per, _) in _SEEN.items()):
        raise ValueError("f1env's driver inputs changed: update drivers.env._SEEN")
    sl = observe.slices(observe.DRIVER_LAYOUT)
    on, gate = np.zeros(OBS, bool), np.arange(OBS)
    for name, (per, cols) in _SEEN.items():
        for first in range(sl[name].start, sl[name].stop, per):
            on[[first + c for c in cols]] = True
            gate[first : first + per] = first
    return on, gate


_NOISY, _GATE = _seen_mask()


def _noise_seen(obs, f):
    """obs x f on the inputs about another car or the safety car that's there (`_SEEN`); everything else exact."""
    return obs * jnp.where(_NOISY & (obs[..., _GATE] < 1.0), f, 1.0)


def _inputs(bank, race, readings, last, flags, cfg: EnvConfig, key=None):
    noise = cfg.sensor_noise > 0 and key is not None
    if noise:
        readings = noisy(readings, key, cfg.sensor_noise)
    obs = observe.driver(bank, race, readings, last, (flags[..., 0], flags[..., 1], flags[..., 2]), cfg.race)
    if not noise:
        return obs
    f = 1.0 + jax.random.uniform(jax.random.fold_in(key, 1), obs.shape, minval=-cfg.sensor_noise, maxval=cfg.sensor_noise)
    return _noise_seen(obs, f)


def _wrap(bank, race, limit, worn, plan, cfg: EnvConfig, pick=None, key=None):
    r, c = race.s.shape
    z = jnp.zeros((r, c), jnp.float32)
    last = RaceActions(z, z, z, z, z)
    flags = jnp.zeros((r, c, 3), bool)
    readings = _lidar(bank, race, cfg)
    pick = all_tracks(bank) if pick is None else jnp.asarray(pick, jnp.int32)
    state = EnvState(race, last, closecalls.barrier_gap(bank, race), flags, worn, limit, jnp.zeros(r, jnp.int32), rewards.new_memory(race.timing.place), plan, pick, jnp.zeros((r, c, 3)))
    return state, _inputs(bank, race, readings, last, flags, cfg, key)


def reset(bank, key, cfg: EnvConfig, tid=None, pick=None):
    race, limit, worn, plan = new_races(bank, key, cfg, tid, pick)
    return _wrap(bank, race, limit, worn, plan, cfg, pick, jax.random.fold_in(key, 7))


def from_race(bank, race, key, cfg: EnvConfig, plan=None):
    """(EnvState, inputs) around a session started elsewhere: f1env's F1Env runs weekends (Q1-Q3 and the grid, 305 km
    races, points). plan: the session's qualifying plan (None in a race). The training principal and its fixed orders
    take over; a race is cut at `time_factor` x its lap-simulation time, a qualifying session 10 minutes after its flag."""
    r, c = race.s.shape
    race = _strong(principal.fixed_orders(race))
    is_q = race.session == QUALIFYING
    limit = jnp.where(is_q, race.session_end + 600.0, cfg.time_factor * bank.lap_time[race.tid] * race.target_laps + 60.0).astype(jnp.float32)
    plan = jnp.full((r, c, cfg.quali_runs), jnp.inf, jnp.float32) if plan is None else jnp.asarray(plan, jnp.float32)
    return _wrap(bank, race, limit, principal.draw_worn(key, (r, c), cfg.worn), plan, cfg, key=jax.random.fold_in(key, 7))


def _restart(bank, s: EnvState, mask, key, cfg: EnvConfig) -> EnvState:
    """New sessions in place of those in `mask` (R,)."""
    race, limit, worn, plan = new_races(bank, key, cfg, pick=s.pick)
    z = jnp.zeros_like(s.last.steer)
    fresh = EnvState(race, RaceActions(z, z, z, z, z), closecalls.barrier_gap(bank, race), jnp.zeros_like(s.flags), worn, limit, jnp.zeros_like(s.tick), rewards.new_memory(race.timing.place), plan, s.pick, jnp.zeros_like(s.order_due))
    return _select(mask, fresh, s)


def restart_races(bank, s: EnvState, obs, mask, key, cfg: EnvConfig):
    """Restart the sessions in mask (R,) now (their track is being swapped out): (state, inputs). The others carry on
    with their inputs unchanged."""
    s = _restart(bank, s, mask, key, cfg)
    readings = _lidar(bank, s.race, cfg)
    fresh = _inputs(bank, s.race, readings, s.last, s.flags, cfg, jax.random.fold_in(key, 7))
    return s, jnp.where(mask[:, None, None], fresh, obs)


def controlled(race):
    """(R, C) the driver drives (or the autopilot, in the pit lane, for a car still in the race)."""
    return (race.status == RACING) & (race.clock >= race.lights_out)[:, None] & ~pt.parked(race.pits)


def race_done(bank, race, limit, cfg: EnvConfig):
    """(R,) nothing left to race: no car running or about to rejoin, past the time limit, or the flag is out and only
    cars that already crashed (and rejoined) are still going. A rejoined car can't finish, so it stops counting once it
    has covered the race distance. With `cfg.learners`, only those cars count as running or rejoining."""
    running = race.status == RACING
    rejoining = cfg.race.respawn & race.dnf & ~race.flags.black & ((race.status == WRECK) | (race.status == OUT))
    laps = jnp.floor(race.line / bank.length[race.tid][:, None])
    through = running & race.dnf & (laps >= race.target_laps[:, None])
    going = (running & ~through) | rejoining
    if cfg.learners:
        going = going & jnp.asarray(cfg.learners, bool)[None, :]
    return ~jnp.any(going, axis=-1) | (race.clock >= limit) | (race.flag & ~jnp.any(running & ~race.dnf, axis=-1))


def random_orders(race: RaceState, due, key, cfg: EnvConfig):
    """Stage C's principal: (race, due). Each head of each learner's orders (pace, team, ERS) is set afresh once the
    race clock passes its `due` time, then again every `order_every` s (drawn): "no order" (pace normal, no team
    order, ERS deploy, as the training principal sets them) for `order_rest` of the draws, else at random push or
    save; hold position, let the team-mate past or defend; ERS save. In races only; f1env's `orders.give` drops
    what isn't possible (a team order with the team-mate out). Pit calls stay the training principal's."""
    r, c = race.s.shape
    k_rest, k_pick, k_next = jax.random.split(key, 3)
    learner = jnp.asarray(cfg.learners, bool) if cfg.learners else jnp.ones(c, bool)
    now = race.clock[:, None, None]
    draw = (now >= due) & ((race.session == RACE)[:, None, None]) & learner[None, :, None] & (race.status == RACING)[..., None]
    rest = jax.random.uniform(k_rest, (r, c, 3)) < cfg.order_rest
    u = jax.random.uniform(k_pick, (r, c, 3))
    pace = jnp.where(rest[..., 0], od.NORMAL, jnp.where(u[..., 0] < 0.5, od.PUSH, od.SAVE))
    team = jnp.where(rest[..., 1], od.NO_ORDER, od.HOLD + jnp.floor(u[..., 1] * 3).astype(jnp.int32))
    ers = jnp.where(rest[..., 2], od.DEPLOY, od.ERS_SAVE)
    new = jnp.stack([pace, team, ers], axis=-1)
    action = jnp.concatenate([jnp.full((r, c, 1), od.KEEP), jnp.where(draw, new, od.KEEP)], axis=-1)
    race, _ = od.give(race, action)
    lo, hi = cfg.order_every
    due = jnp.where(draw, now + jax.random.uniform(k_next, (r, c, 3), minval=lo, maxval=hi), due)
    return race, due


def step(bank, s: EnvState, action, key, cfg: EnvConfig, events: bool = False):
    """One 20 Hz step. events=True also returns f1env's RaceEvents (the viewer records from them)."""
    k_worn, k_reset, k_noise = jax.random.split(key, 3)
    acts = to_race_actions(action)
    race = s.race
    decide = s.tick % PRINCIPAL_EVERY == 0
    is_q = race.session == QUALIFYING
    call = principal.calls(bank, race, s.worn_at, cfg.race, cfg.two_compound_km)
    race = pit_orders(race, jnp.where((decide & ~is_q)[:, None], call, -1))
    order_due = s.order_due
    if cfg.random_orders:  # stage C: team orders, at random, for the learners in races
        race, order_due = random_orders(race, s.order_due, jax.random.fold_in(key, 11), cfg)
    if cfg.quali or cfg.quali_share > 0:  # qualifying: f1env's default runs (out-lap, flying lap, in-lap, twice)
        runs = quali_orders(bank, race, s.plan, cfg.race)
        race = race._replace(pits=_select(decide & is_q, runs.pits, race.pits))
    live = ~race_done(bank, race, s.limit, cfg)  # without auto-reset (evals) a race that's over stays over
    valid = controlled(race) & live[:, None]

    new, ev = race_step(bank, race, acts, cfg.race)
    new = jax.tree.map(lambda a, b: a.astype(b.dtype), new, race)
    stopped = rewards.standing_still(new, cfg.race.blocked_gap, cfg.race.queue_speed)
    pace = jnp.maximum(jnp.minimum(bank.lap_speed[race.tid[:, None], race.idx], ev.sc_limit), 1.0)  # the reference pace
    rated = jnp.maximum(ev.progress, 0.0) / (jnp.asarray(cfg.race.tyres.life_km)[race.tyres.compound] * 1000.0)
    since = race.clock - race.lights_out  # a standing start's first seconds (a spread start is past them)
    launching = ((race.session == RACE) & (since >= 0.0) & (since < rewards.LAUNCH_S))[:, None] & (race.status == RACING)
    gap = closecalls.barrier_gap(bank, new)
    sc_gap = rewards.safety_car_gap(bank, new, cfg.race.control)  # the queue's first car: behind the safety car, not beside it
    amounts, mem = rewards.amounts(
        ev, acts, s.last, s.mem, valid, ev.respawned, quali=(is_q, race.best_lap, new.best_lap), stopped=stopped, pace=pace, rated_wear=rated,
        sector_len=jnp.broadcast_to(bank.length[race.tid][:, None] / 3.0, valid.shape), close=rewards.too_close(new, ev.gap_ahead, sc_gap),
        warned_before=race.flags.warnings,
        per_second={
            "launch": rewards.launch_throttle(launching, acts),
            "car_close": closecalls.cars(new), "wall_close": closecalls.walls(gap, s.wall_gap, new.status == RACING),
            "overspeed": rewards.over_limit(new, ev.sc_limit), "sc_pass": rewards.alongside(bank, new, cfg.race.control, sc_gap), "blue": new.flags.blue,
        },
        per_step={"pull_away": rewards.pull_away(race, new), "waiting": rewards.waiting(bank, race, new, cfg.race), **(rewards.order_steps(race, ev, acts, STEP_S) if cfg.random_orders else {})},
        left=rewards.race_left(bank, race, cfg.crash_left_km) if cfg.crash_left_km else None,
    )
    terminal = valid & (ev.crashed | ev.finished | ev.black_flag)
    over = race_done(bank, new, s.limit, cfg)
    end = jnp.where(terminal, TERMINAL, jnp.where(valid & over[:, None], CUT, CONTINUE)).astype(jnp.int8)
    worn_at = jnp.where(ev.pit_fitted, principal.draw_worn(k_worn, s.worn_at.shape, cfg.worn), s.worn_at)
    flags = jnp.stack([ev.drs_available, ev.drs_open, ev.ers_on], axis=-1)
    nxt = EnvState(new, acts, gap, flags, worn_at, s.limit, s.tick + 1, mem, s.plan, s.pick, order_due)

    if cfg.auto_reset:
        nxt = jax.lax.cond(jnp.any(over), lambda: _restart(bank, nxt, over, k_reset, cfg), lambda: nxt)
    readings = _lidar(bank, nxt.race, cfg)
    obs = _inputs(bank, nxt.race, readings, nxt.last, nxt.flags, cfg, k_noise)

    ref = bank.lap_time[s.race.tid][:, None]
    info = {
        "amounts": jnp.sum(amounts, axis=(0, 1)),
        "valid": jnp.sum(valid),
        "crashes": jnp.sum(valid & ev.crashed),
        "finishes": jnp.sum(valid & ev.finished),
        "black_flags": jnp.sum(valid & ev.black_flag),
        "hit": jnp.sum(valid & ev.crashed & ev.crash_car & (ev.crash_blame < 0.5)),
        "laps": jnp.sum(valid & ev.lap_done),
        "laps_counted": jnp.sum(valid & ev.lap_counted),
        "lap_ratio": jnp.sum(jnp.where(valid & ev.lap_counted, ev.lap_time / ref, 0.0)),
        "pit_stops": jnp.sum(valid & ev.pit_in),
        "dead_tyre": jnp.sum(valid & ev.tyre_dead),
        "races": jnp.sum(over & live),
        "metres": jnp.sum(jnp.where(valid & ~ev.off_track, ev.progress, 0.0)),
        "wear": jnp.sum(jnp.where(valid, ev.tyre_wear * 100.0, 0.0)),
    }
    info = {k: v.astype(jnp.float32) for k, v in info.items()}
    dyn = ev.dynamics
    slipping = (dyn.wheelspin > SLIPPING) | (dyn.lockup_front > SLIPPING) | (dyn.lockup_rear > SLIPPING)
    kinds = warning_kinds(jnp.where(valid, ev.warned, 0))
    out = Out(
        amounts, valid, end, valid & ev.lap_done, valid & ev.records.under_attack, jnp.where(valid, ev.records.lost, 0), valid & slipping,
        jnp.where(valid & ~ev.off_track, ev.progress, 0.0), jnp.where(valid, ev.tyre_wear * 100.0, 0.0), valid & ev.crashed,
        valid & ev.crashed & ev.crash_car & (ev.crash_blame < 0.5), kinds, kinds * (valid & ev.black_flag)[..., None],
        *clean_running(ev, valid, pace, launching, new.flags.blue),
        jnp.where(valid, ev.records.gained, 0),
    )
    return (nxt, obs, out, info, ev) if events else (nxt, obs, out, info)


@lru_cache(maxsize=32)
def compiled(name: str, cfg: EnvConfig):
    """reset / from_race / race_done / restart_races for one config, jit-compiled once and reused."""
    return jax.jit(partial({"reset": reset, "from_race": from_race, "race_done": race_done, "restart_races": restart_races}[name], cfg=cfg))
