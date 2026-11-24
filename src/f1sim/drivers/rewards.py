"""The drivers' rewards (Phase 2, section 4), built from each step's f1env events.

Every term is measured in its own unit (metres, crashes, seconds, % of tyre life ...), then weighted:
reward = sum over terms of amount x base value x the profile's multiplier for the term's style (common terms 1x).
The amounts are kept per term so each one's share of a profile's reward can be logged.

| Term | Amount per step | Base | Style |
|---|---|---|---|
| progress | steps' worth of reference pace: metres along the centreline / (the lap simulation's speed there, or the speed limit if lower, x 0.05 s), while the car's centre is on the track; under the safety car only up to the limit (speeding earns nothing more) | +1 (a step at reference pace) | common |
| crash | crashes (ends the car's race: no respawn), the full cost whoever was to blame: a car hit by another pays it too | -2500 | common |
| black_flag | disqualified (ends the car's race): a 4th warning, or at once for passing the safety car or keeping a place gained under it for 5 s (f1env, since 2026-11-15) | -2500 | common |
| sector | each third of a lap driven on the track this race (paid once per third, so going back and forth doesn't farm it) | +10 | common |
| sc_close | seconds closer than 0.5 s behind the car ahead, or behind the safety car itself, while the safety car is called or out (queues that squeeze up end in chains of rear-ends) | -5 | common |
| overspeed | seconds x m/s over the safety car's speed limit (from the limit itself, not from the warning's +5%), while racing on the track | -20 | common |
| sc_pass | seconds x m alongside the car ahead with no passing in force (the safety car out, to the restart), or alongside the safety car: from nose level with its tail (0) to level (5.6, the safety car's black flag) or a metre past its nose (6.6, a car: the place is then owed) | -20 | common |
| warning | warnings 1-3, each by its number: the 1st counts 0.25, the 2nd 0.5, the 3rd 0.75 (the 4th is the black flag) | -2500 | common |
| bump | contacts under 20 km/h | -2 | common |
| pit | entering the pit lane on a call | +5 | common |
| dead_tyre | seconds on a tyre at 0% life | -2 | common |
| finish | finishing the race | +50 | common |
| time | steps racing in a race (not qualifying, not under the red flag): a fixed charge per step, charged at the end of each ~200 m mini-sector for the steps it took (and when the car's race ends), so a faster car pays less, soon enough for the critics to see it (a charge every step didn't make run middle-pace1 faster, 2026-11-21); finishing sooner pays. Progress alone pays about the same over a whole race at any speed (less per step, more steps), and Middle stage B run sc2 drove 17% slower than the lap sim alone on the track (x1.115, 191 km per tyre set; b6c x0.953, 123 km) (since 2026-11-20) | 0 (Middle -0.35, Aggressive -0.4, Defensive -0.3; halved 2026-11-24, stage A 0) | common |
| off_track | seconds with the car's centre off the track (run-off, gravel) | -2 | common |
| standstill | seconds stopped (under 1 m/s) on the track, unless queuing behind a moving car | -20 | common |
| pull_away | change of speed within the stopped band (0-1 m/s) this step, racing on the track out of the pit lane both before and after it: paid for speeding up out of it, charged the same for slowing into it, so a stop and a start come to nothing and it can't be farmed (since 2026-11-16) | +100 per m/s | common |
| waiting | change of the waiting deposit this step: a car standing still (under 1 m/s, racing on the road) pays it in as its wait level rises (`f1env.race.wait_level`: most of it in the first 8 s without progress), and gets it back as it drives off; a car whose race ends while it waits has lost it (since 2026-11-16) | +2000 per whole deposit | common |
| order_pace | stage C: counted mini-sectors pushed (f1env's push gain faster than the car's own reference) or saved (20% less tyre wear) as ordered, less those missed (`order_steps`; 0 without the random principal, since 2026-11-24) | +20 | common |
| order_team | stage C: a "let past" failed (the team-mate within 1 s behind for 3 corners) or the team-mate passed under "hold" | -500 | common |
| order_defend | stage C: places lost on the track under "defend" (on top of place_lost) | -300 | common |
| order_ers | stage C: seconds deploying ERS under "save", or not deploying under "deploy" at full throttle with battery left | -5 per second | common |
| overtake | places gained on the track and held for 2 s (none from the safety car's call to the restart) | +20 | overtake |
| place_lost | places lost on the track | -25 | place_lost |
| holding | seconds keeping the place with a rival within 1 s behind | +1 | holding |
| inside_line | seconds on the inside of the next corner, rival within 1 s behind, before the braking point | +2 | inside_line |
| tyre_wear | % of tyre life used beyond the compound's rated wear for the distance | -10 | tyre_wear |
| tyre_save | % of tyre life saved against the compound's rated wear for the distance (the other side of tyre_wear) | 0 (Smooth: +60) | common |
| blue | seconds shown the blue flag: the leader within 1 s behind a car about to be lapped, or a place gained under the safety car not yet given back (since 2026-11-15) | -10 | common |
| inputs | change of steer + throttle + brake | -0.05 | smoothness |
| slip | wheelspin + lock-ups (how far past the grip) + sliding (how far each axle is past its tyres' peak slip angle) | -0.5 | smoothness |
| quali_place | qualifying: change in standing points, 21 for P1 down to 0 for P22 (no time: 0) | +1 | common |
| quali_q2 | qualifying: getting into (+1) or dropping out of (-1) the top 16, the places that move on to Q2 | +3 | common |
| quali_q3 | qualifying: the same for the top 10, the places that move on to Q3 | +5 | common |
| quali_lap | qualifying: a new personal best lap (the first time set counts) | +3 | common |
| launch | the first 2 s after the lights go out (a standing start): seconds of throttle, counted only with the brake fully off | +20 | common |
| car_close | seconds at a car close call's level (0..1: the time to contact with another car, wrecks too, under 0.4 s; `closecalls`) | -20 | car_close |
| wall_close | seconds at a barrier close call's level (the same, to the nearest barrier) | -20 | wall_close |

Qualifying is paid as the standing changes, not at the flag: a car gets a place's points when its lap puts it there and
loses them when another car's lap pushes it down while it's out on track, so over a session it collects what its final
place is worth, close to the lap that earned it (the flag is minutes away, far beyond any profile's horizon).

An overtake pays only for places better than the best one already paid this episode, so two cars can't farm points
by swapping places (for Aggressive an overtake is worth 2.7x a place lost). The f1env autopilot takes every car with a
committed call into the pit lane, so a called stop can't be missed: only the "+" side of the pit term exists.

Standing still costs at once: drivers learned to wait in the pack at a race start (throttle and brakes off) until the
cars around them had gone, which the stuck rule only punishes after 6 s. A car queuing behind a moving car (within 30 m,
over 1 m/s: f1env's stuck-rule queue), in the pit lane or held at the lights isn't charged.

The blue flag costs by the second (since 2026-11-15): until then only the warning every 3 corners did, and Smooth,
slower by design and lapped often, went from 3 to 20-42 disqualifications for ignoring it per eval in run s5 without
learning to move over. A car that lets the leader by at once pays a few seconds of it.

Progress is paid against the reference pace at that point, the lap simulation's speed (the safety car's limit when
it's lower), so a step at reference pace pays +1 in a hairpin, on a straight or behind the safety car alike: the reward
measures how well a car drives where it is, and a safety car period isn't a stretch of low reward the driver can't
change. (Until 2026-10-31 it paid per metre, +1 for a step at the 310 km/h top speed: ~0.6 on a clean lap, ~0.4 behind
the safety car.) Tyre wear is charged only beyond what the compound's rated life allows for the distance, so the cost
is for wearing tyres harder than nominal, not a tax on every metre. `tyre_save` (since 2026-11-14) is the other side:
it pays for wear under the rated amount, and only Smooth has it (+60 per %, the rate its tyre_wear costs: one line
through the rated wear), so Smooth gains from every bit of life saved without a cost on every metre, which would
lower every step's reward towards that of standing still. It is paid per metre driven, at most 80% of the rated wear
(a fifth of the wear comes with distance alone): ~3% of a step at reference pace at +20, ~9% at +60.

The launch term (since 2026-11-06) pays for a flat-out standing start: in the 2 s after the lights go out, each
second of throttle with the brake fully off earns +20 (+40 at most), so drivers leave the grid instead of waiting in
the pack for the cars around them to go (v2 simply forced 2 s of full throttle). A spread start or a qualifying session
doesn't pay it. Close calls (`closecalls`, after v2's) charge the steps before a crash, not only the crash: a car or a
barrier closer than 0.4 s at the current closing speed costs up to -20 a second (-1 a step at full level), weighted by
the profile's `car_close` and `wall_close` styles.

A crash or a black flag costs -2500 (since 2026-11-05; the crash was -100), far more than anything a car can gain
over its ~8 s horizon: at -100 the drivers rated +1% of pace as worth ~6 points more crash risk per lap, and crashed on
15-25% of laps. Every car in a contact crash pays the full cost, the one hit too, so drivers keep out of trouble even
when it costs a place. The black flag costs as much as a crash, so it's never a cheaper way out (it was the 4th
warning's -50). Since 2026-11-06 the warnings before it cost 25%, 50% and 75% of a black flag (they were -50 each):
almost every black flag was for speeding under the safety car, and with the first three warnings free the drivers
got no signal until the 4th, so a disqualification now costs 250% of a crash in all.

The overspeed term (since 2026-11-11) charges every step over the safety car's limit, by how far over: 1 m/s (3.6
km/h) over for a second costs -20, as much as a second of driving at reference pace earns, and 5% over the 232 km/h
catch-up limit (3.2 m/s) for the 1 s it takes to earn a warning costs ~-64 before the warning's -625. The warnings alone
didn't teach the drivers to hold the limit: b3 (2026-11-11, progress already capped at the limit) still got ~0.5
speeding warnings a lap after 600k steps, no fewer than at its start. A warning is a rare, large cost arriving a second
after the throttle that caused it, with nothing saying 240 km/h is worse than 235 or that 230 is right; this term
gives that slope at every step, and since it starts at the limit itself the drivers have a margin under the warning's
5%. The stuck rule (6 s stopped) and hitting the safety car count as crashes.

The sc_pass term (since 2026-11-13) does the same for passing with the safety car out: every step a car is alongside
the car ahead of it in the order costs by how far alongside, from its centre a car's length behind that car's (0) to
a metre ahead (6.6 m, where the place changes hands and the warning comes). Fully alongside for a second costs -112,
so backing out is always cheaper than completing the pass (the warning's -625 on top). b4c's passing warnings never
fell (11 per eval at its 1.8M best, 100-200 after 3.0M) with the warning as the only signal. Free where passing is:
a lapped car unlapping itself, a car stopped, crawling or under the minimum speed, cars in the pit lane.

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
from f1env import racecontrol as rc
from f1env.pits import OFF
from f1env.race import RACE, RACING, wait_level

TOP_SPEED = 86.1  # m/s, the stock car's (310 km/h)
TERMS = ("progress", "crash", "warning", "bump", "pit", "dead_tyre", "finish", "off_track", "overtake", "place_lost", "holding", "inside_line", "tyre_wear", "inputs", "slip", "quali_place", "quali_q2", "quali_q3", "quali_lap", "standstill", "sector", "sc_close", "black_flag", "launch", "car_close", "wall_close", "overspeed", "sc_pass", "tyre_save", "blue", "pull_away", "waiting", "time", "order_pace", "order_team", "order_defend", "order_ers")
STYLE = {"overtake": "overtake", "place_lost": "place_lost", "holding": "holding", "inside_line": "inside_line", "tyre_wear": "tyre_wear", "inputs": "smoothness", "slip": "smoothness", "car_close": "car_close", "wall_close": "wall_close"}
STYLES = ("overtake", "place_lost", "holding", "inside_line", "tyre_wear", "smoothness", "car_close", "wall_close")

BASE = {
    "progress": 1.0,
    "crash": -2500.0,
    "warning": -2500.0,  # x 0.25 / 0.5 / 0.75 for the 1st / 2nd / 3rd warning (`warning_shares`)
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
    "launch": 20.0,  # per second of throttle in the 2 s after lights out, brake off: +40 for a flat-out start
    "car_close": -20.0,  # per second at full level (-1 a step, as much as a step at reference pace earns)
    "wall_close": -20.0,
    "overspeed": -20.0,  # per m/s over the safety car's limit, per second
    "sc_pass": -20.0,  # per m alongside the car ahead with no passing in force, per second (`alongside`)
    "blue": -10.0,  # per second under the blue flag (being lapped, or owing a place gained under the safety car)
    "pull_away": 100.0,  # per m/s of speed gained within the stopped band (lost: the same back), `pull_away`
    "waiting": 2000.0,  # the whole waiting deposit (-200 scaled, paid within ~8 s of standing; back on driving off), `waiting`
    "tyre_save": 0.0,  # per % of tyre life saved against the rated wear: only Smooth sets it (+60, as its tyre_wear costs)
    # stage C, the principal's orders (2026-11-24, `order_steps`; 0 without orders): obeying matters, but never more than staying safe
    "order_pace": 20.0,  # per mini-sector pushed or saved as ordered (missed: the same off)
    "order_team": -500.0,  # per let-past failed (3 corners) or pass of the team-mate under "hold"
    "order_defend": -300.0,  # per place lost on the track under "defend" (on top of place_lost)
    "order_ers": -5.0,  # per second deploying under "save", or not deploying under "deploy" with the battery and full throttle
    "time": 0.0,  # per step racing in a race, charged at each mini-sector's end: the profiles that race for pace set it ([rewards] time), Smooth doesn't
}
LAUNCH_S = 2.0  # s after the lights go out that the launch term pays (v2 forced full throttle for 2 s)
HOLD_S = 2.0  # an overtake counts once the place has been held this long
STOPPED_SPEED = 1.0  # m/s: slower than this counts as standing still
Q2_PLACES, Q3_PLACES = 16, 10  # the places that move on (f1env's Q1 and Q2 knock out 6 each)


def _band(r):
    """(R, C) each car's speed clipped to the stopped band (0 to STOPPED_SPEED)."""
    return jnp.clip(jnp.hypot(r.cars.vx, r.cars.vy), 0.0, STOPPED_SPEED)


def _on_road_both(before, after):
    """(R, C) racing and out of the pit lane both before and after the step."""
    return (before.status == RACING) & (before.pits.stage == OFF) & (after.status == RACING) & (after.pits.stage == OFF)


def pull_away(before, after):
    """(R, C) m/s of speed gained within the stopped band (0 to STOPPED_SPEED) in the step, negative for speed lost
    into it; 0 unless the car is racing on the track, out of the pit lane (the autopilot drives there), both before
    and after the step. Why (2026-11-16): Smooth parked when the safety car was called, its critics rating one step of
    throttle from rest a little below staying put (-49.5 against -46.3, 92% of stopped steps): standing still costs
    the same whatever the car does in a step, and one step of throttle (~0.5 m/s) leaves it under 1 m/s, so nothing
    told it to move. This pays the first step at once; a stop and a start add up to 0."""
    return jnp.where(_on_road_both(before, after), _band(after) - _band(before), 0.0)


def waiting(bank, before, after, cfg):
    """(R, C) change of the waiting deposit in the step, as a share of the whole deposit: negative while a car pays it
    in, positive as it gets it back; 0 unless the car is racing on the road both before and after the step (so a car
    whose race ends while it waits has lost what it paid). The share a car holds is its wait level (f1env's
    `wait_level`: 1 - exp(-seconds without progress / 4), from its stuck timer) while it stands still, fading to none
    as its speed rises through the stopped band (0 to STOPPED_SPEED).

    Why (2026-11-16): at a crash scene with the safety car called Smooth pulled off the track and waited, at 3 scenes
    in 4, and a quarter of those waits ended in the stuck time-out. Its critics rated waiting off the track at -74 and
    driving past the wreck at -158 (what followed: -16 and -21), and could not see the time-out: it is up to 90 s
    away, their view ~8 s, and the timer was not an input. Over the next 2 minutes a wait cost ~200 against driving
    on. So the deposit is that cost, paid at the start of the wait where the critics can see it (-50 a second at
    first, 64% of it inside their view), and handed back in full the moment the car is going again: leaving a wait
    pays at once, a short stop costs nothing in the end, and a stop and a start add up to 0, so it can't be farmed.
    cfg: the RaceConfig the race runs under (the stuck times)."""

    def held(r):
        return wait_level(bank, r, cfg) * (1.0 - _band(r) / STOPPED_SPEED)

    return jnp.where(_on_road_both(before, after), held(before) - held(after), 0.0)


def standing_still(race, blocked_gap: float, queue_speed: float):
    """(R, C) on the track (racing, out of the pit lane), under STOPPED_SPEED and not queuing behind a moving car (one
    within `blocked_gap` m ahead in the path going faster than `queue_speed`, as f1env's stuck rule has it)."""
    from f1env.race import traffic_ahead

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
    clock: jnp.ndarray  # (R, C) steps racing (the time term) since the last mini-sector ended, not yet charged


def new_memory(place):
    place = jnp.asarray(place, jnp.int32)
    return Memory(jnp.zeros_like(place), jnp.zeros_like(place), place, jnp.zeros(place.shape, jnp.float32), jnp.zeros_like(place), jnp.zeros(place.shape, jnp.float32))


def weights(base: dict | None = None, styles: dict | None = None, scale: float = 1.0) -> np.ndarray:
    """(terms,) base value x the profile's multiplier for each term's style, x `scale` (the reward scale)."""
    base = {**BASE, **(base or {})}
    styles = styles or {}
    return np.array([scale * (base[t] * styles.get(STYLE[t], 1.0) if t in STYLE else base[t]) for t in TERMS], np.float32)


def _bits(x, n: int = 7):  # f1env's 7 warning kinds (blend since 2026-11-12; close and far since 2026-11-19, counted from 2026-11-20)
    return sum((x >> k) & 1 for k in range(n))


SC_GAP_S = 0.5  # under the safety car, closer than this (s at the car's speed) behind the car ahead is too close


LEFT_CAP = 8.0  # `race_left`: a crash costs at most 9x its base (-2,250 scaled)


def race_left(bank, race, per_km: float):
    """(R, C) the extra share of a crash's or a black flag's cost for the race it ends early: `per_km` for each km
    the car still had to drive to the flag (0 in qualifying, or with per_km 0). A crash costs the critics -250 and
    the ~13 of reward they see ahead (gamma 0.994, ~8 s), but it ends the car's race: ~30 a km of pay lost, ~1,500
    in an average race (run s8 2026-11-17: per step the 400k best and the crashier Middle it started from earned
    the same, 0.0768 against 0.0767; only staying in the race paid). Charging a share of it at the crash lets the
    critics see it: a lap-1 crash costs far more than one on the last lap. At most LEFT_CAP more (x9 in all, -2,250
    scaled), so a long race's lap-1 crash doesn't swamp the critics' targets."""
    total = (race.target_laps.astype(jnp.float32) * bank.length[race.tid])[:, None]
    km = jnp.maximum(total - race.line, 0.0) / 1000.0
    return jnp.where((race.session == RACE)[:, None], jnp.minimum(km * per_km, LEFT_CAP), 0.0)


def safety_car_gap(bank, race, cp=None):
    """(R, C) m along the track from each car's centre to the safety car's ahead of it (negative: the car is past it),
    while the safety car is on the track and solid; inf otherwise, for a car in the pit lane or out, for a lapped car
    (it may pass the safety car) and beyond 200 m. For `too_close` and `alongside`: the first car of the queue has no
    car ahead of it, only the safety car."""
    from f1env import pits as pt

    cp = rc.ControlParams() if cp is None else cp
    total = bank.length[race.tid][:, None]
    d = jnp.mod(race.sc.s[:, None] - race.s + total / 2, total) - total / 2
    racing = race.status == RACING
    there = (rc.on_track(race.sc) & (race.sc.age >= cp.sc_solid_after))[:, None] & racing & (race.pits.stage == pt.OFF)
    free = rc.lapped(bank, race.tid, race.line, racing)
    return jnp.where(there & ~free & (jnp.abs(d) < 200.0), d, jnp.inf)


def too_close(race, gap_ahead, sc_gap=None):
    """(R, C) with the safety car called, out or just in (to the restart), the gap to the car ahead in the path is under
    SC_GAP_S at the car's speed (moving at over 10 m/s). sc_gap: `safety_car_gap` (None: cars only): the same behind
    the safety car itself, body to body, wherever the car is across the track (since 2026-11-15)."""
    from f1env.sensors import BODY_LENGTH

    speed = jnp.hypot(race.cars.vx, race.cars.vy)
    period = (race.sc.phase != rc.NONE) & (race.sc.phase != rc.RED)
    close = gap_ahead < SC_GAP_S * speed
    if sc_gap is not None:
        close = close | (sc_gap - BODY_LENGTH < SC_GAP_S * speed)
    return period[:, None] & (speed > 10.0) & close


def over_limit(race, limit):
    """(R, C) m/s a racing car is over its speed limit after the step (`limit`: f1env's sc_limit, inf without one)."""

    speed = jnp.hypot(race.cars.vx, race.cars.vy)
    return jnp.where((race.status == RACING) & jnp.isfinite(limit), jnp.maximum(speed - limit, 0.0), 0.0)


def alongside(bank, race, cp=None, sc_gap=None):
    """(R, C) m a racing car is alongside the car ahead of it in the order while passing is forbidden (the safety car
    on the track, to the restart; the red flag): 0 with its centre a car's length behind that car's, up to
    BODY_LENGTH + pass_margin once it's that far ahead (the passing warning's point). The deepest one if several.
    0 where passing is allowed: a lapped car, a car ahead that is stopped or crawling,
    either car in the pit lane or out. cp: f1env's ControlParams (default: its own). sc_gap: `safety_car_gap` (None:
    cars only): the same alongside the safety car, 0 a car's length behind it, BODY_LENGTH level with it (the black
    flag's point). Since 2026-11-15: the first car of the queue rode beside the safety car (within 8 m of it 30% of the
    time, at no cost) and was past it whenever it braked for a corner, 13 of 17 passing warnings in 7 races."""
    from f1env import pits as pt
    from f1env.sensors import BODY_LENGTH

    cp = rc.ControlParams() if cp is None else cp
    racing = race.status == RACING
    on = racing & (race.pits.stage == pt.OFF)
    speed = jnp.hypot(race.cars.vx, race.cars.vy)
    lead = race.line[:, :, None] - race.line[:, None, :]  # [i, j]: i ahead of j by this
    behind = jnp.swapaxes(race.flags.ahead, 1, 2)  # [i, j]: j is ahead of i in the order
    passable = (on & (speed > cp.pass_slow))[:, None, :]  # j counts
    depth = jnp.clip(lead + BODY_LENGTH, 0.0, BODY_LENGTH + cp.pass_margin)
    deepest = jnp.max(jnp.where(behind & passable & on[:, :, None], depth, 0.0), axis=-1)
    free = rc.lapped(bank, race.tid, race.line, racing)
    cars = jnp.where(rc.no_passing(race.sc)[:, None] & ~free, deepest, 0.0)
    if sc_gap is None:
        return cars
    return jnp.maximum(cars, jnp.where(jnp.isfinite(sc_gap), jnp.clip(BODY_LENGTH - sc_gap, 0.0, BODY_LENGTH + cp.pass_margin), 0.0))


MAX_WARNINGS = 4  # the 4th disqualifies (f1env's race control)


def warning_shares(warned, before):
    """(R, C) the warning term's amount: each warning given this step (`warned` bits) counts its number / 4, the 1st
    0.25, the 2nd 0.5, the 3rd 0.75; the 4th is the black flag, paid by its own term. before: warnings before the step."""
    n = _bits(warned)
    total = jnp.zeros(jnp.shape(n), jnp.float32)
    for j in range(MAX_WARNINGS):
        k = before + 1 + j
        total = total + jnp.where((j < n) & (k < MAX_WARNINGS), k / MAX_WARNINGS, 0.0)
    return total


def launch_throttle(launching, actions):
    """(R, C) the launch term's rate: the throttle within LAUNCH_S of the lights going out at a standing start
    (`launching`), 0 with any brake."""
    return launching * jnp.where(actions.brake > 0, 0.0, actions.throttle)


PER_SECOND = ("launch", "car_close", "wall_close", "overspeed", "sc_pass", "blue")  # `amounts`' per_second terms: (R, C) rates, x dt
ORDER_TERMS = ("order_pace", "order_team", "order_defend", "order_ers")  # stage C's (`order_steps`)
PER_STEP = ("pull_away", "waiting", *ORDER_TERMS)  # ... and its per_step ones: (R, C) amounts as they are


def order_steps(race, events, actions, dt: float) -> dict:
    """Stage C: the order terms' amounts this step (`amounts`' per_step) under the orders in force before it, from
    f1env's compliance records. order_pace: +1 a counted mini-sector pushed (at least f1env's push gain faster than its
    reference) or saved (20% less wear) as ordered, -1 one missed (slower than its reference / no wear saved);
    order_team: a let-past failed (the team-mate within 1 s behind for 3 corners) or the team-mate passed under
    "hold"; order_defend: places lost on the track under "defend"; order_ers: seconds deploying under "save", or
    not deploying under "deploy" at full throttle with battery left (ERS allowed)."""
    from f1env import orders as od
    from f1env.pits import OFF

    rec, o = events.records, race.orders
    road = (race.status == RACING) & (race.pits.stage == OFF)
    pace = rec.push_met.astype(jnp.float32) + rec.save_met - rec.push_missed - rec.save_missed
    team = rec.let_past_failed.astype(jnp.float32) + rec.passed_mate_on_hold
    defend = jnp.where(road & (o.team == od.DEFEND), rec.lost, 0).astype(jnp.float32)
    lazy = (o.ers == od.DEPLOY) & ~events.ers_on & (actions.throttle > 0.9) & (race.battery > 0.5) & race.ers_enabled[:, None]
    ers = (road & (rec.ers_on_save | lazy)) * dt
    return {"order_pace": pace, "order_team": team, "order_defend": defend, "order_ers": ers}


def amounts(events, actions, last_actions, mem: Memory, valid, respawned, dt: float = 0.05, quali=None, stopped=None, pace=None, rated_wear=None, sector_len=None, close=None, warned_before=None, per_second=None, per_step=None, left=None):
    """(R, C, terms) the amount of each term this step for cars the driver controlled before it, and the new memory.
    actions / last_actions: f1env RaceActions sent this step and the one before. quali: (is_qualifying (R,), best laps
    before the step, after it (R, C)) for the qualifying terms (None: none). stopped: (R, C) standing still on the
    track and not queuing (`standing_still`; None: never). pace: (R, C) the reference speed, m/s (None: the 86.1 m/s
    top speed); rated_wear: (R, C) the tyre life this step's distance uses at the compound's rated life (None: 0);
    sector_len: (R, C) a third of the lap, m (None: no sector term); close: (R, C) under 0.5 s behind the car ahead with
    the safety car called or out (`too_close`; None: never). warned_before: (R, C) each car's warnings before the step
    (None: none). per_second: {term: (R, C)} the PER_SECOND terms' rates, each paid x dt: launch (the throttle
    within LAUNCH_S of the lights going out at a standing start, 0 on the brake), car_close and wall_close (the close-call
    levels 0..1, `closecalls`), overspeed (m/s over the safety car's limit, `over_limit`), sc_pass (m alongside the car
    ahead with no passing in force, `alongside`), blue (shown the blue flag); per_step: {term: (R, C)} the PER_STEP
    terms as they are: pull_away (m/s gained within the stopped band, `pull_away`), waiting (the share of the waiting
    deposit handed back, negative for paid in, `waiting`). A term left out is 0. left: (R, C) the extra share of a
    crash's or a black flag's cost for the race it ends early (`race_left`; None: 0)."""
    rec = events.records
    dyn = events.dynamics

    # overtakes: a place gained is paid once held for HOLD_S, and only beyond the best place already paid; losing a
    # place cancels a pending gain first. From the safety car's call to the restart places gained pay nothing: they
    # don't count, and the best place moves up with them so they're never paid later
    no_sc = (events.sc_phase == rc.NONE)[:, None]
    pending = mem.pending + jnp.where(no_sc, rec.gained, 0)
    cancel = jnp.minimum(pending, rec.lost)
    pending = pending - cancel
    lost = rec.lost - cancel
    timer = jnp.where(rec.gained > 0, round(HOLD_S / dt), mem.timer - 1)
    due = (timer <= 0) & (pending > 0)
    paid = jnp.where(due, jnp.clip(mem.best - rec.place, 0, pending), 0)
    best = jnp.where(due, jnp.minimum(mem.best, rec.place), mem.best)
    pending = jnp.where(due, 0, pending)
    best = jnp.where(no_sc, best, jnp.minimum(best, rec.place))
    pending = jnp.where(no_sc, pending, 0)
    best = jnp.where(respawned, rec.place, best)  # a new episode
    # sectors: each third of a lap of progress on the track this episode, paid once (the car's own distance, so a red
    # flag moving it into the pit lane or back to the grid gives none)
    on_track = jnp.where(events.off_track, 0.0, events.progress)
    # under the safety car progress pays up to the speed limit: going faster than the limit earns nothing more
    paid_progress = jnp.minimum(on_track, jnp.where(jnp.isfinite(events.sc_limit), events.sc_limit * dt, jnp.inf))
    driven = jnp.where(respawned, 0.0, mem.driven + jnp.where(valid, on_track, 0.0))
    done = jnp.zeros_like(mem.sectors) if sector_len is None else jnp.floor(driven / sector_len).astype(jnp.int32)
    sector_paid = jnp.maximum(done - mem.sectors, 0)
    mem = Memory(pending, timer, best, driven, jnp.where(respawned, 0, jnp.maximum(mem.sectors, done)), mem.clock)

    inputs = jnp.abs(actions.steer - last_actions.steer) + jnp.abs(actions.throttle - last_actions.throttle) + jnp.abs(actions.brake - last_actions.brake)
    slip = dyn.wheelspin + dyn.lockup_front + dyn.lockup_rear + dyn.slide_front + dyn.slide_rear  # drifting costs too
    hit = 1.0 if left is None else 1.0 + left  # the full cost, plus a share for the race it ends early (`race_left`)
    parts = {
        "progress": paid_progress / ((TOP_SPEED if pace is None else pace) * dt),  # no progress off the track: cutting doesn't pay
        # the full cost, whoever was to blame, plus a share for the race it ends early (`race_left`)
        "crash": events.crashed * hit,
        # a black flag given at once (passing the safety car, a place kept under it) comes with its warning kind: the
        # black_flag term pays it, not this one
        "warning": jnp.where(events.black_flag, 0.0, warning_shares(events.warned, 0 if warned_before is None else warned_before)),
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
    parts["black_flag"] = events.black_flag * hit
    per_second, per_step = per_second or {}, per_step or {}
    assert set(per_second) <= set(PER_SECOND) and set(per_step) <= set(PER_STEP), (sorted(per_second), sorted(per_step))
    parts.update({k: per_second[k] * dt if k in per_second else zero for k in PER_SECOND})
    parts.update({k: per_step.get(k, zero) for k in PER_STEP})
    # saving tyres pays only in racing: behind the safety car every car saves them, and the pay for it kept Smooth's
    # queue under the minimum speed (slow warnings 80-106 an eval in run s8 at 2.5-2.8M, against 5-8 for Middle)
    parts["tyre_save"] = zero if rated_wear is None else jnp.where(no_sc, jnp.maximum(rated_wear - events.tyre_wear, 0.0) * 100.0, 0.0)
    red = (events.sc_phase == rc.RED)[:, None]
    in_race = jnp.ones_like(red) if quali is None else ~quali[0][:, None]
    # charged per mini-sector (~200 m, ~3.5 s at race pace), for the steps it took: a faster car pays less, sooner, which
    # the critics (~8 s ahead) can see; a charge per step is the same whatever the speed, so it couldn't make Middle
    # faster (run middle-pace1, 2026-11-21). A race's total is the same. The rest is charged when the car's race ends
    ticks = jnp.where(respawned, 0.0, mem.clock) + jnp.where(in_race & ~red & valid, 1.0, 0.0) + zero
    ms_due = rec.ms_done | events.crashed | events.finished | events.black_flag
    parts["time"] = jnp.where(ms_due, ticks, 0.0)
    mem = mem._replace(clock=jnp.where(ms_due, 0.0, ticks))
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
