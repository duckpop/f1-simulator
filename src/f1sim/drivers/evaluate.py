"""Evals (Phase 2, section 5), on the fixed eval tracks with the default car (no upgrades), the training principal, no
respawn and each actor's mean action.

- **Alone** (`alone`): one car per race; the share of the scheduled distance completed, the reward per step under
  the profile's own weights (with its style terms, so it separates the profiles), the finish rate, crashes per lap,
  lap time (against each track's lap simulation), tyre stints and km per set of tyres.
- **Mixed**: stage B's eval is a full weekend per eval track (`season.eval_weekend`), the profiles against a fixed
  reference field; `better_ordered` and `summary_mixed` rank and print its per-profile results.
- **Incidents per 100 km**: crashes and black flags per 100 km driven on the track, with a 95% confidence interval
  (Poisson), alone and in the weekend race.
- **Picking the best** (`better`, `better_ordered`): continuous measures, so one crash doesn't decide it: in stage B
  first fewer incidents per 100 km in the race, when the difference is beyond the noise (`safer`); else the distance
  completed (more than 2% apart); then the points (more than 0.1 per race apart); then the profile's reward per step. Evals
  of another version (before 2026-11-05: EVAL_VERSION 1 and 2, other rewards and measures) aren't compared.

    f1sim-eval --checkpoint runs/middle/stage-a/best.npz --tracks tracks --laps 50

v2 to beat (alone, 50 laps): finished 53%, crashes 1.39% per lap.
"""

from __future__ import annotations

import argparse
import math
import os
from functools import lru_cache

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.race import RaceConfig
from f1env.weekend import score

from .. import checkpoint, tqc
from . import env as de
from .config import PROFILES
from .rewards import TERMS

EVAL_VERSION = 4  # 2026-11-24: one-lap qualifying in the weekend (3: 2026-11-05, incidents per 100 km first; rewards x0.1)
DISTANCE_BAND = 0.02  # within this share of the distance completed, the reward per step decides
CLEAN_PACE_BAND = 0.006  # stage B, a profile judged on its pace: its clean race pace more than this share faster than the best's, against the reference field's (since 2026-11-21): the reference against itself read -0.37% and -0.78% on two seeds (runs/clean_pace_check.py)
TYRE_BAND = 0.05  # stage B, a profile judged on its tyres (Smooth): more than this share more km per set of tyres wins
POINTS_BAND = 0.1  # stage B: within this many points per race (and 2% of the distance), the reward decides
Z95 = 1.96


def per_100km(count: float, km: float) -> tuple[float, float, float]:
    """(rate, low, high): events per 100 km and a 95% interval for it (Byar's approximation to the exact Poisson one)."""
    km = float(km)  # plain floats: numpy ones don't go into the JSON log
    if km <= 0:
        return float("nan"), float("nan"), float("nan")
    k = float(count)
    lo = 0.0 if k == 0 else k * (1 - 1 / (9 * k) - Z95 / (3 * math.sqrt(k))) ** 3
    k1 = k + 1
    hi = k1 * (1 - 1 / (9 * k1) + Z95 / (3 * math.sqrt(k1))) ** 3
    f = 100.0 / km
    return k * f, lo * f, hi * f


def incidents(count: float, km: float) -> dict:
    rate, lo, hi = per_100km(count, km)
    return {"incidents": float(count), "km": float(km), "incidents_per_100km": rate, "incidents_ci": [lo, hi]}


def safer(a: dict, b: dict) -> int:
    """-1: a has fewer incidents per km than b beyond the noise (a two-sided 95% test on the two Poisson rates), 1:
    more, 0: within the noise (or not measured)."""
    if not all(k in r for r in (a, b) for k in ("incidents", "km")) or a["km"] <= 0 or b["km"] <= 0:
        return 0
    ka, kb, ma, mb = a["incidents"], b["incidents"], a["km"], b["km"]
    se = math.sqrt((ka + 0.5) / ma**2 + (kb + 0.5) / mb**2)
    z = (ka / ma - kb / mb) / se
    return -1 if z < -Z95 else (1 if z > Z95 else 0)


def incidents_text(r: dict) -> str:
    if not np.isfinite(r.get("incidents_per_100km", np.nan)):
        return ""
    lo, hi = r["incidents_ci"]
    return f"incidents {r['incidents_per_100km']:.2f}/100 km [{lo:.2f}-{hi:.2f}] ({r['incidents']:.0f} in {r['km']:.0f} km)  "


def eval_config(n_races: int, laps: int) -> de.EnvConfig:
    return de.EnvConfig(races=n_races, cars=1, fixed_laps=laps, upgrades=False, random_grid=False, auto_reset=False, race=RaceConfig(respawn=False))


@lru_cache(maxsize=8)
def _runner(ecfg: de.EnvConfig, cfg: tqc.TQCConfig, steps: int):
    @jax.jit
    def run(bank, actor, state, obs, key):
        def body(c, k):
            state, obs = c
            action = tqc.act(actor, cfg, obs.reshape(-1, de.OBS), k, mean=True)
            state, obs, _, info = de.step(bank, state, action.reshape(ecfg.races, ecfg.cars, de.ACTIONS), k, ecfg)
            return (state, obs), info

        keys = jax.random.split(key, steps)
        zero = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), jax.eval_shape(lambda: body((state, obs), keys[0])[1]))

        def going(c):  # blocks left, and a race still on
            i, state, _, _ = c
            return (i < steps // de.EXACT_END) & ~jnp.all(de.race_done(bank, state.race, state.limit, ecfg))

        def block(c):  # the block's steps, their info summed
            i, state, obs, total = c
            (state, obs), info = jax.lax.scan(body, (state, obs), jax.lax.dynamic_slice_in_dim(keys, i * de.EXACT_END, de.EXACT_END))
            return i + 1, state, obs, jax.tree.map(lambda a, b: a + jnp.sum(b, axis=0), total, info)

        i, state, obs, info = jax.lax.while_loop(going, block, (jnp.int32(0), state, obs, zero))
        return state, obs, info, i * de.EXACT_END

    return run


def alone(bank, actor, cfg: tqc.TQCConfig, laps: int, seeds: int = 1, key=None, chunk: int = 2000, max_steps: int = 2_000_000, weights=None) -> dict:
    """Every track in the bank `seeds` times, `laps` laps each. weights: the profile's reward weights (terms,), for
    its reward per step (None: not measured)."""
    key = jax.random.PRNGKey(12345) if key is None else key
    n = int(bank.length.shape[0])
    tid = jnp.tile(jnp.arange(n), seeds)
    ecfg = eval_config(n * seeds, laps)
    k_reset, key = jax.random.split(key)
    state, obs = de.compiled("reset", ecfg)(bank, k_reset, tid=tid)
    run = _runner(ecfg, cfg, chunk)
    done_fn = de.compiled("race_done", ecfg)
    totals: dict[str, float] = {}
    steps = 0
    while steps < max_steps:
        key, k = jax.random.split(key)
        state, obs, info, ran = run(bank, actor, state, obs, k)
        for name, v in info.items():  # summed over the chunk's steps
            if name == "amounts":
                per_term = np.asarray(v)
                totals["off_track_s"] = totals.get("off_track_s", 0.0) + float(per_term[TERMS.index("off_track")])
                if weights is not None:
                    totals["reward"] = totals.get("reward", 0.0) + float(per_term @ np.asarray(weights))
            else:
                totals[name] = totals.get(name, 0.0) + float(np.asarray(v).sum())
        steps += int(ran)
        if int(ran) < chunk or bool(np.asarray(done_fn(bank, state.race, state.limit)).all()):
            break
    laps_done = totals.get("laps", 0.0)
    stops = totals.get("pit_stops", 0.0)
    return {
        "eval_version": EVAL_VERSION,
        "races": n * seeds,
        "laps": laps_done,
        "distance": min(laps_done / (n * seeds * laps), 1.0),
        "reward_per_step": totals.get("reward", 0.0) / max(totals.get("valid", 0.0), 1.0) if weights is not None else float("nan"),
        "km_per_tyre_set": totals.get("metres", 0.0) / 1000.0 / max(totals.get("wear", 0.0) / 100.0, 1e-9),
        "finish_rate": totals.get("finishes", 0.0) / (n * seeds),
        "crashes_per_lap": totals.get("crashes", 0.0) / max(laps_done, 1.0),
        "lap_ratio": totals.get("lap_ratio", 0.0) / max(totals.get("laps_counted", 0.0), 1.0),
        "laps_per_stint": laps_done / max(n * seeds + stops, 1.0),
        "dead_tyre_s": totals.get("dead_tyre", 0.0) * 0.05,
        "off_track_share": totals.get("off_track_s", 0.0) / max(totals.get("valid", 0.0) * 0.05, 1e-9),
        "steps": steps,
        **incidents(totals.get("crashes", 0.0) + totals.get("black_flags", 0.0), totals.get("metres", 0.0) / 1000.0),
    }


def _version(r: dict | None) -> int:
    return int(r.get("eval_version", 1)) if r else 0


def _distance_then_reward(a: dict, b: dict) -> bool:
    if abs(a["distance"] - b["distance"]) > DISTANCE_BAND:
        return a["distance"] > b["distance"]
    return a["reward_per_step"] > b["reward_per_step"]


def better(a: dict, b: dict | None) -> bool:
    """Is eval a better than b: more than 2% more of the scheduled distance completed, else (within 2% of it) more
    reward per step under the profile's own weights. An older eval (another version) is no bar."""
    if b is None or _version(b) != _version(a):
        return True
    return _distance_then_reward(a, b)


def summary(r: dict) -> str:
    crashes = f"crashes {100 * r['crashes_per_lap']:.2f}%/lap" if r["laps"] >= 1 else f"{r['crashes_per_lap']:.0f} crashes, no lap done"
    new = f"distance {100 * r['distance']:.1f}%  reward {r['reward_per_step']:.3f}/step  " if "distance" in r else ""
    if np.isfinite(r.get("clean_pace", np.nan)):
        vs = f" ({100 * (r['pace_vs_reference'] - 1):+.2f}% vs reference{', riskier' if r.get('riskier_than_reference') else ''})" if np.isfinite(r.get("pace_vs_reference", np.nan)) else ""
        new += f"clean pace x{r['clean_pace']:.3f} of the lap sim{vs}  "

    tyres = f"  {r['km_per_tyre_set']:.0f} km/set" if "km_per_tyre_set" in r else ""
    return f"{incidents_text(r)}{new}finished {100 * r['finish_rate']:.0f}%  {crashes}  lap x{r['lap_ratio']:.3f} of the lap sim  off track {100 * r.get('off_track_share', 0.0):.1f}%  {r['laps_per_stint']:.1f} laps/stint{tyres}  ({r['laps']:.0f} laps)"


def riskier(r: dict, k: float = 1.0) -> bool | None:
    """More incidents per km than k x the reference field's in the same races, beyond the noise (the `safer` test;
    None: no reference)."""
    if "reference_incidents" not in r or r.get("km", 0) <= 0 or r["reference_km"] <= 0:
        return None
    ka, kb, ma, mb = r["incidents"], k * r["reference_incidents"], r["km"], r["reference_km"]
    se = math.sqrt((ka + 0.5) / ma**2 + k * k * (r["reference_incidents"] + 0.5) / mb**2)
    return (ka / ma - kb / mb) / se > Z95


ORDER_BAND = 0.10  # stage C: an order score more than this share better (and over 1 a 100 km) wins
BANDS = {"pace": CLEAN_PACE_BAND, "attack_success": 0.03, "attacks_held": 0.02}  # within these, the next check decides
DEFAULT_ORDER = ("incidents_vs_best", "distance", "points")


def _step(name: str, a: dict, b: dict) -> bool | None:
    """One check of a best order: True / False when it decides (a better / worse), None when it passes it on."""
    if name == "incidents" or name.startswith("cap"):  # not riskier than k x the reference field (incidents: 1 x)
        k = float(name.split(":")[1]) if ":" in name else 1.0
        ra, rb = riskier(a, k), riskier(b, k)
        return None if ra is None or rb is None or ra == rb else rb
    if name == "incidents_vs_best":  # fewer incidents than the best beyond the noise (across two evals)
        side = safer(a, b)
        return None if side == 0 else side < 0
    if name == "pace":  # clean race pace against the reference's: lower is faster
        pa, pb = a.get("pace_vs_reference", np.nan), b.get("pace_vs_reference", np.nan)
        return None if not (np.isfinite(pa) and np.isfinite(pb)) or abs(pa - pb) <= BANDS["pace"] else pa < pb
    if name in ("attack_success", "attacks_held"):  # shares: higher is better
        pa, pb = a.get(name, np.nan), b.get(name, np.nan)
        return None if not (np.isfinite(pa) and np.isfinite(pb)) or abs(pa - pb) <= BANDS[name] else pa > pb
    if name == "tyres":
        ta, tb = a.get("km_per_tyre_set"), b.get("km_per_tyre_set")
        return None if not (ta and tb) or abs(ta - tb) <= TYRE_BAND * tb else ta > tb
    if name == "orders":  # stage C: the order terms' cost per 100 km (`season` order_score, higher is better), 10% apart
        sa, sb_ = a.get("order_score"), b.get("order_score")
        if sa is None or sb_ is None or abs(sa - sb_) <= max(ORDER_BAND * abs(sb_), 1.0):
            return None
        return sa > sb_
    if name == "distance":
        return None if abs(a["distance"] - b["distance"]) <= DISTANCE_BAND else a["distance"] > b["distance"]
    if name == "points":
        return None if abs(a["points_per_race"] - b["points_per_race"]) <= POINTS_BAND else a["points_per_race"] > b["points_per_race"]
    raise ValueError(f"unknown best check {name!r}")


def better_ordered(a: dict, b: dict | None, order=DEFAULT_ORDER) -> bool:
    """Stages B-D: a profile's own checks in its order (`[stage_b] best_order`, since
    2026-11-21), the first that decides wins, else the reward per step. An older eval (another version) is no bar.
    incidents / cap:k: not riskier than (k x) the reference field in the same races; incidents_vs_best: fewer than the
    best beyond the noise; pace: clean race pace against the reference's (0.6%); attack_success, attacks_held: the
    shares (3, 2 points); tyres: km per set (5%); distance (2%); points (0.1 a race)."""
    if b is None or _version(b) != _version(a):
        return True
    for name in order:
        side = _step(name, a, b)
        if side is not None:
            return side
    return a["reward_per_step"] > b["reward_per_step"]


def summary_mixed(r: dict) -> str:
    crashes = f"crashes {100 * r['crashes_per_lap']:.2f}%/lap" if r["laps"] >= 1 else f"{r['crashes_per_lap']:.0f} crashes, no lap done"
    held = f"{100 * r['attacks_held']:.0f}%" if np.isfinite(r.get("attacks_held", np.nan)) else "-"
    new = f"distance {100 * r['distance']:.1f}%  reward {r['reward_per_step']:.3f}/step  " if "distance" in r else ""
    if np.isfinite(r.get("clean_pace", np.nan)):
        vs = f" ({100 * (r['pace_vs_reference'] - 1):+.2f}% vs reference{', riskier' if r.get('riskier_than_reference') else ''})" if np.isfinite(r.get("pace_vs_reference", np.nan)) else ""
        new += f"clean pace x{r['clean_pace']:.3f} of the lap sim{vs}  "

    return (
        f"{r['points_per_race']:5.2f} pts/race  {incidents_text(r)}{new}finished {100 * r['finish_rate']:3.0f}%  out on lap 1 {100 * r['out_lap1']:3.0f}%  "
        f"{crashes}  off track {100 * r.get('off_track_share', 0.0):.1f}%  overtakes {r['overtakes_per_race']:.2f}  lost {r['places_lost_per_race']:.2f}  attacks held {held}"
        + warnings_text(r)
        + orders_text(r)
    )


def orders_text(r: dict) -> str:
    """Stage C: how the drivers took their orders, per 100 km ("" with no orders given)."""
    o = r.get("orders") or {}
    if not any(o.values()):
        return ""
    return f"\n{'':18}orders per 100 km: pace {o['pace']:+.1f} mini-sectors (met - missed)  team broken {o['team']:.2f}  places lost defending {o['defend']:.2f}  ERS against {o['ers']:.1f} s  (score {r['order_score']:+.0f})"


def warnings_text(r: dict) -> str:
    """The race's incidents split up: crashes, black flags by the kind of the 4th warning, and every warning by kind
    (speeding, passing, blue, slow, blend, close, far). Empty for an eval from before 2026-11-11."""
    if "black_flags_by" not in r:
        return ""
    kinds = lambda d: " ".join(f"{k} {v:.0f}" for k, v in d.items())
    return f"\n                  out: {r['crashes']:.0f} crashes, {r['black_flags']:.0f} black flags ({kinds(r['black_flags_by'])})  warnings: {kinds(r['warnings_by'])}"


def main(argv=None):
    from f1env.sensors import make_bank

    from .config import load
    from .tracks import eval_tracks

    ap = argparse.ArgumentParser(description="Evaluate a driver alone on the fixed eval tracks.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tracks", help="track pool directory (f1env-tracks); default: generate the eval tracks")
    ap.add_argument("--width", default="normal", help='"normal" (12-15 m), or a wide copy: 26 or 20')
    ap.add_argument("--laps", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args(argv)
    info = checkpoint.meta(a.checkpoint)
    s = load(info["profile"], info.get("preset", "laptop"))
    template = tqc.init(jax.random.PRNGKey(0), s.tqc)
    agent, _ = checkpoint.load(a.checkpoint, template)
    bank = make_bank(eval_tracks(a.tracks, a.width))
    r = alone(bank, agent.actor, s.tqc, a.laps, a.seeds, weights=s.weights)
    print(f"{info['profile']} ({a.checkpoint}): {summary(r)}")
    print("v2 middle alone, 50 laps: finished 53%, crashes 1.39%/lap")


if __name__ == "__main__":
    main()
