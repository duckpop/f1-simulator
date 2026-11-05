"""The drivers' rewards (Phase 2, section 4), built from each step's f1env events.

Every term is measured in its own unit (metres, crashes, seconds, % of tyre life ...), then weighted:
reward = sum over terms of amount x base value x the profile's multiplier for the term's style (common terms 1x).
The amounts are kept per term so each one's share of a profile's reward can be logged.

| Term | Amount per step | Base | Style |
|---|---|---|---|
| progress | steps' worth of reference pace: metres along the centreline / (the lap simulation's speed there, or the speed limit if lower, x 0.05 s), while the car's centre is on the track | +1 (a step at reference pace) | common |
| crash | crashes (ends the car's race: no respawn), the full cost whoever was to blame: a car hit by another pays it too | -2500 | common |
| black_flag | disqualified for a 4th warning (ends the car's race) | -2500 | common |
| sector | each third of a lap driven on the track this race (paid once per third, so going back and forth doesn't farm it) | +10 | common |
| sc_close | seconds closer than 0.5 s behind the car ahead while the safety car is called or out (queues that squeeze up end in chains of rear-ends) | -5 | common |
| warning | warnings (the 4th is a black flag, which ends it) | -50 | common |
| bump | contacts under 20 km/h | -2 | common |
| pit | entering the pit lane on a call | +5 | common |
| dead_tyre | seconds on a tyre at 0% life | -2 | common |
| finish | finishing the race | +50 | common |
| off_track | seconds with the car's centre off the track (run-off, gravel) | -2 | common |
| standstill | seconds stopped (under 1 m/s) on the track, unless queuing behind a moving car | -20 | common |
| overtake | places gained on the track and held for 2 s | +20 | overtake |
| place_lost | places lost on the track | -25 | place_lost |
| holding | seconds keeping the place with a rival within 1 s behind | +1 | holding |
| inside_line | seconds on the inside of the next corner, rival within 1 s behind, before the braking point | +2 | inside_line |
| tyre_wear | % of tyre life used beyond the compound's rated wear for the distance | -10 | tyre_wear |
| inputs | change of steer + throttle + brake | -0.05 | smoothness |
| slip | wheelspin + lock-ups (how far past the grip) + sliding (how far each axle is past its tyres' peak slip angle) | -0.5 | smoothness |
| quali_place | qualifying: change in standing points, 21 for P1 down to 0 for P22 (no time: 0) | +1 | common |
| quali_q2 | qualifying: getting into (+1) or dropping out of (-1) the top 16, the places that move on to Q2 | +3 | common |
| quali_q3 | qualifying: the same for the top 10, the places that move on to Q3 | +5 | common |
| quali_lap | qualifying: a new personal best lap (the first time set counts) | +3 | common |

Qualifying is paid as the standing changes, not at the flag: a car gets a place's points when its lap puts it there and
loses them when another car's lap pushes it down while it's out on track, so over a session it collects what its final
place is worth, close to the lap that earned it (the flag is minutes away, far beyond any profile's horizon).

An overtake pays only for places better than the best one already paid this episode, so two cars can't farm points
by swapping places (for Aggressive an overtake is worth 2.7x a place lost). The f1env autopilot takes every car with a
committed call into the pit lane, so a called stop can't be missed: only the "+" side of the pit term exists.

Standing still costs at once: drivers learned to wait in the pack at a race start (throttle and brakes off) until the
cars around them had gone, which the stuck rule only punishes after 6 s. A car queuing behind a moving car (within 30 m,
over 1 m/s: f1env's stuck-rule queue), in the pit lane or held at the lights isn't charged.

Progress is paid against the reference pace at that point, the lap simulation's speed (the safety car's limit when
it's lower), so a step at reference pace pays +1 in a hairpin, on a straight or behind the safety car alike: the reward
measures how well a car drives where it is, and a safety car period isn't a stretch of low reward the driver can't
change. (Until 2026-10-31 it paid per metre, +1 for a step at the 310 km/h top speed: ~0.6 on a clean lap, ~0.4 behind
the safety car.) Tyre wear is charged only beyond what the compound's rated life allows for the distance, so the cost
is for wearing tyres harder than nominal, not a tax on every metre.

A crash or a black flag costs -2500 (since 2026-11-05; the crash was -100), far more than anything a car can gain
over its ~8 s horizon: at -100 the drivers rated +1% of pace as worth ~6 points more crash risk per lap, and crashed on
15-25% of laps. Every car in a contact crash pays the full cost, the one hit too, so drivers keep out of trouble even
when it costs a place. The black flag costs as much as a crash, so it's never a cheaper way out (it was the 4th
warning's -50). The stuck rule (6 s stopped) and hitting the safety car count as crashes.

The weighted sum is multiplied by `scale` ([rewards] in drivers.toml, 0.1), so the critics' targets stay around
-250 to +12 while the base values read in their own units.

f1env has no track limits (run-off only has less grip), and progress is measured along the centreline, so a driver can
gain by cutting a corner across the run-off; Middle learned to cut the inside of esses. So progress made with the car's
centre off the track doesn't count (cutting 60 m at speed forfeits ~14, worth far more than the time it saves), and
every second off the track costs a little more.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

TOP_SPEED = 86.1  # m/s, the stock car's (310 km/h)
TERMS = ("progress", "crash", "warning", "bump", "pit", "dead_tyre", "finish", "off_track", "overtake", "place_lost", "holding", "inside_line", "tyre_wear", "inputs", "slip", "quali_place", "quali_q2", "quali_q3", "quali_lap", "standstill", "sector", "sc_close", "black_flag")
STYLE = {"overtake": "overtake", "place_lost": "place_lost", "holding": "holding", "inside_line": "inside_line", "tyre_wear": "tyre_wear", "inputs": "smoothness", "slip": "smoothness"}
STYLES = ("overtake", "place_lost", "holding", "inside_line", "tyre_wear", "smoothness")

BASE = {
    "progress": 1.0,
    "crash": -2500.0,
    "warning": -50.0,
    "bump": -2.0,
    "pit": 5.0,
    "dead_tyre": -2.0,
    "finish": 50.0,
    "off_track": -2.0,
    "overtake": 20.0,
    "place_lost": -25.0,
    "holding": 1.0,
    "inside_line": 2.0,
    "tyre_wear": -10.0,
    "inputs": -0.05,
    "slip": -0.5,
    "quali_place": 1.0,
    "quali_q2": 3.0,
    "quali_q3": 5.0,
    "quali_lap": 3.0,
    "standstill": -20.0,
    "sector": 10.0,
    "sc_close": -5.0,
    "black_flag": -2500.0,
}
HOLD_S = 2.0  # an overtake counts once the place has been held this long
STOPPED_SPEED = 1.0  # m/s: slower than this counts as standing still
Q2_PLACES, Q3_PLACES = 16, 10  # the places that move on (f1env's Q1 and Q2 knock out 6 each)


def standing_still(race, blocked_gap: float, queue_speed: float):
    """(R, C) on the track (racing, out of the pit lane), under STOPPED_SPEED and not queuing behind a moving car (one
    within `blocked_gap` m ahead in the path going faster than `queue_speed`, as f1env's stuck rule has it)."""
    from f1env.pits import OFF
    from f1env.race import RACING, traffic_ahead

    racing = race.status == RACING
    gap, v_ahead = traffic_ahead(race.cars, racing)
    queuing = (gap <= blocked_gap) & (v_ahead > queue_speed)
    slow = jnp.hypot(race.cars.vx, race.cars.vy) < STOPPED_SPEED
    return racing & (race.pits.stage == OFF) & slow & ~queuing


def quali_standing(best_lap):
    """(R, C) best laps (inf: no time) -> (points, in the top 16, in the top 10). Points: cars - place, so 21 for P1
    down to 0 for P22 with 22 cars; no time: 0 points, in neither group. Equal times: the lower car number ahead."""
    n = best_lap.shape[-1]
    timed = jnp.isfinite(best_lap)
    b_i, b_j = best_lap[..., :, None], best_lap[..., None, :]
    order = jnp.arange(n)
    faster = timed[..., None, :] & ((b_j < b_i) | ((b_j == b_i) & (order[None, :] < order[:, None])))
    place = 1 + jnp.sum(faster, axis=-1)
    points = jnp.where(timed, n - place, 0)
    return points, timed & (place <= Q2_PLACES), timed & (place <= Q3_PLACES)


class Memory(NamedTuple):
    """Per car, carried between steps."""

    pending: jnp.ndarray  # (R, C) places gained, waiting to be held for HOLD_S
    timer: jnp.ndarray  # (R, C) steps left until they're paid
    best: jnp.ndarray  # (R, C) the best place an overtake has been paid for this episode
    driven: jnp.ndarray  # (R, C) m of progress on the track this episode (the sector term)
    sectors: jnp.ndarray  # (R, C) thirds of a lap paid this episode


def new_memory(place):
    place = jnp.asarray(place, jnp.int32)
    return Memory(jnp.zeros_like(place), jnp.zeros_like(place), place, jnp.zeros(place.shape, jnp.float32), jnp.zeros_like(place))


def weights(base: dict | None = None, styles: dict | None = None, scale: float = 1.0) -> np.ndarray:
    """(terms,) base value x the profile's multiplier for each term's style, x `scale` (the reward scale)."""
    base = {**BASE, **(base or {})}
    styles = styles or {}
    return np.array([scale * (base[t] * styles.get(STYLE[t], 1.0) if t in STYLE else base[t]) for t in TERMS], np.float32)


def _bits(x, n: int = 4):
    return sum((x >> k) & 1 for k in range(n))


SC_GAP_S = 0.5  # under the safety car, closer than this (s at the car's speed) behind the car ahead is too close


def too_close(race, gap_ahead):
    """(R, C) with the safety car called, out or just in (to the restart), the gap to the car ahead in the path is under
    SC_GAP_S at the car's speed (moving at over 10 m/s)."""
    from f1env import racecontrol as rc

    speed = jnp.hypot(race.cars.vx, race.cars.vy)
    period = (race.sc.phase != rc.NONE) & (race.sc.phase != rc.RED)
    return period[:, None] & (speed > 10.0) & (gap_ahead < SC_GAP_S * speed)


def amounts(events, actions, last_actions, mem: Memory, valid, respawned, dt: float = 0.05, quali=None, stopped=None, pace=None, rated_wear=None, sector_len=None, close=None):
    """(R, C, terms) the amount of each term this step for cars the driver controlled before it, and the new memory.
    actions / last_actions: f1env RaceActions sent this step and the one before. quali: (is_qualifying (R,), best laps
    before the step, after it (R, C)) for the qualifying terms (None: none). stopped: (R, C) standing still on the
    track and not queuing (`standing_still`; None: never). pace: (R, C) the reference speed, m/s (None: the 86.1 m/s
    top speed); rated_wear: (R, C) the tyre life this step's distance uses at the compound's rated life (None: 0);
    sector_len: (R, C) a third of the lap, m (None: no sector term); close: (R, C) under 0.5 s behind the car ahead with
    the safety car called or out (`too_close`; None: never)."""
    rec = events.records
    dyn = events.dynamics

    # overtakes: a place gained is paid once held for HOLD_S, and only beyond the best place already paid; losing a
    # place cancels a pending gain first
    pending = mem.pending + rec.gained
    cancel = jnp.minimum(pending, rec.lost)
    pending = pending - cancel
    lost = rec.lost - cancel
    timer = jnp.where(rec.gained > 0, round(HOLD_S / dt), mem.timer - 1)
    due = (timer <= 0) & (pending > 0)
    paid = jnp.where(due, jnp.clip(mem.best - rec.place, 0, pending), 0)
    best = jnp.where(due, jnp.minimum(mem.best, rec.place), mem.best)
    pending = jnp.where(due, 0, pending)
    best = jnp.where(respawned, rec.place, best)  # a new episode
    # sectors: each third of a lap of progress on the track this episode, paid once (the car's own distance, so a red
    # flag moving it into the pit lane or back to the grid gives none)
    on_track = jnp.where(events.off_track, 0.0, events.progress)
    driven = jnp.where(respawned, 0.0, mem.driven + jnp.where(valid, on_track, 0.0))
    done = jnp.zeros_like(mem.sectors) if sector_len is None else jnp.floor(driven / sector_len).astype(jnp.int32)
    sector_paid = jnp.maximum(done - mem.sectors, 0)
    mem = Memory(pending, timer, best, driven, jnp.where(respawned, 0, jnp.maximum(mem.sectors, done)))

    inputs = jnp.abs(actions.steer - last_actions.steer) + jnp.abs(actions.throttle - last_actions.throttle) + jnp.abs(actions.brake - last_actions.brake)
    slip = dyn.wheelspin + dyn.lockup_front + dyn.lockup_rear + dyn.slide_front + dyn.slide_rear  # drifting costs too
    parts = {
        "progress": on_track / ((TOP_SPEED if pace is None else pace) * dt),  # no progress off the track: cutting doesn't pay
        "crash": events.crashed,  # the full cost, whoever was to blame
        "warning": _bits(events.warned),
        "bump": events.bump,
        "pit": events.pit_in,
        "dead_tyre": events.tyre_dead * dt,
        "off_track": events.off_track * dt,
        "finish": events.finished,
        "overtake": paid,
        "place_lost": lost,
        "holding": (rec.under_attack & (rec.lost == 0)) * dt,
        "inside_line": rec.inside_line * dt,
        "tyre_wear": jnp.maximum(events.tyre_wear - (0.0 if rated_wear is None else rated_wear), 0.0) * 100.0,
        "inputs": inputs,
        "slip": slip,
    }
    zero = jnp.zeros_like(slip)
    parts["standstill"] = zero if stopped is None else stopped * dt
    parts["sector"] = sector_paid
    parts["sc_close"] = zero if close is None else close * dt
    parts["black_flag"] = events.black_flag
    if quali is None:
        parts.update(quali_place=zero, quali_q2=zero, quali_q3=zero, quali_lap=zero)
    else:
        is_q, best_old, best_new = quali
        (p0, q2_0, q3_0), (p1, q2_1, q3_1) = quali_standing(best_old), quali_standing(best_new)
        on = is_q[:, None]
        parts.update(
            quali_place=jnp.where(on, p1 - p0, 0),
            quali_q2=jnp.where(on, q2_1.astype(jnp.int32) - q2_0, 0),
            quali_q3=jnp.where(on, q3_1.astype(jnp.int32) - q3_0, 0),
            quali_lap=on & jnp.isfinite(best_new) & (best_new < best_old),
        )
    out = jnp.stack([jnp.asarray(parts[t], jnp.float32) for t in TERMS], axis=-1)
    return jnp.where(valid[..., None], out, 0.0), mem
