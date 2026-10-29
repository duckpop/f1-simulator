"""The drivers' rewards (Phase 2, section 4), built from each step's f1env events.

Every term is measured in its own unit (metres, crashes, seconds, % of tyre life ...), then weighted:
reward = sum over terms of amount x base value x the profile's multiplier for the term's style (common terms 1x).
The amounts are kept per term so each one's share of a profile's reward can be logged.

| Term | Amount per step | Base | Style |
|---|---|---|---|
| progress | metres along the centreline, while the car's centre is on the track | +1 for a step at top speed (as v2) | common |
| crash | crashes (ends the episode) | -70 | common |
| warning | warnings (the 4th is a black flag, which ends it) | -50 | common |
| bump | contacts under 20 km/h | -2 | common |
| pit | entering the pit lane on a call | +5 | common |
| dead_tyre | seconds on a tyre at 0% life | -2 | common |
| finish | finishing the race | +50 | common |
| off_track | seconds with the car's centre off the track (run-off, gravel) | -2 | common |
| overtake | places gained on the track and held for 2 s | +20 | overtake |
| place_lost | places lost on the track | -20 | place_lost |
| holding | seconds keeping the place with a rival within 1 s behind | +1 | holding |
| inside_line | seconds on the inside of the next corner, rival within 1 s behind, before the braking point | +2 | inside_line |
| tyre_wear | % of tyre life used | -10 | tyre_wear |
| inputs | change of steer + throttle + brake | -0.05 | smoothness |
| slip | wheelspin + lock-ups (how far past the grip) | -0.2 | smoothness |

An overtake pays only for places better than the best one already paid this episode, so two cars can't farm points
by swapping places (for Aggressive an overtake is worth 4x a place lost). The f1env autopilot takes every car with a
committed call into the pit lane, so a called stop can't be missed: only the "+" side of the pit term exists.

f1env has no track limits (run-off only has less grip), and progress is measured along the centreline, so a driver can
gain by cutting a corner across the run-off; Middle learned to cut the inside of esses. So progress made with the car's
centre off the track doesn't count (cutting 60 m at speed forfeits ~14, worth far more than the time it saves), and
every second off the track costs a little more.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

TERMS = ("progress", "crash", "warning", "bump", "pit", "dead_tyre", "finish", "off_track", "overtake", "place_lost", "holding", "inside_line", "tyre_wear", "inputs", "slip")
STYLE = {"overtake": "overtake", "place_lost": "place_lost", "holding": "holding", "inside_line": "inside_line", "tyre_wear": "tyre_wear", "inputs": "smoothness", "slip": "smoothness"}
STYLES = ("overtake", "place_lost", "holding", "inside_line", "tyre_wear", "smoothness")

BASE = {
    "progress": 1.0 / (86.1 * 0.05),
    "crash": -70.0,
    "warning": -50.0,
    "bump": -2.0,
    "pit": 5.0,
    "dead_tyre": -2.0,
    "finish": 50.0,
    "off_track": -2.0,
    "overtake": 20.0,
    "place_lost": -20.0,
    "holding": 1.0,
    "inside_line": 2.0,
    "tyre_wear": -10.0,
    "inputs": -0.05,
    "slip": -0.2,
}
HOLD_S = 2.0  # an overtake counts once the place has been held this long


class Memory(NamedTuple):
    """Per car, carried between steps."""

    pending: jnp.ndarray  # (R, C) places gained, waiting to be held for HOLD_S
    timer: jnp.ndarray  # (R, C) steps left until they're paid
    best: jnp.ndarray  # (R, C) the best place an overtake has been paid for this episode


def new_memory(place):
    place = jnp.asarray(place, jnp.int32)
    return Memory(jnp.zeros_like(place), jnp.zeros_like(place), place)


def weights(base: dict | None = None, styles: dict | None = None) -> np.ndarray:
    """(terms,) base value x the profile's multiplier for each term's style."""
    base = {**BASE, **(base or {})}
    styles = styles or {}
    return np.array([base[t] * styles.get(STYLE[t], 1.0) if t in STYLE else base[t] for t in TERMS], np.float32)


def _bits(x, n: int = 3):
    return sum((x >> k) & 1 for k in range(n))


def amounts(events, actions, last_actions, mem: Memory, valid, respawned, dt: float = 0.05):
    """(R, C, terms) the amount of each term this step for cars the driver controlled before it, and the new memory.
    actions / last_actions: f1env RaceActions sent this step and the one before."""
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
    mem = Memory(pending, timer, best)

    inputs = jnp.abs(actions.steer - last_actions.steer) + jnp.abs(actions.throttle - last_actions.throttle) + jnp.abs(actions.brake - last_actions.brake)
    slip = dyn.wheelspin + dyn.lockup_front + dyn.lockup_rear
    parts = {
        "progress": jnp.where(events.off_track, 0.0, events.progress),  # no progress off the track: cutting doesn't pay
        "crash": events.crashed,
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
        "tyre_wear": events.tyre_wear * 100.0,
        "inputs": inputs,
        "slip": slip,
    }
    out = jnp.stack([jnp.asarray(parts[t], jnp.float32) for t in TERMS], axis=-1)
    return jnp.where(valid[..., None], out, 0.0), mem
