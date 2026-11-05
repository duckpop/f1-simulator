"""Evals (Phase 2, section 5), on the fixed eval tracks with the default car (no upgrades), the training principal, no
respawn and each actor's mean action.

- **Alone** (`alone`): one car per race; the share of the scheduled distance completed, the reward per step under
  the profile's own weights (with its style terms, so it separates the profiles), the finish rate, crashes per lap,
  lap time (against each track's lap simulation), tyre stints and km per set of tyres.
- **Mixed**: stage B's eval is a full weekend per eval track (`season.eval_weekend`), the profiles against a fixed
  reference field; `better_mixed` and `summary_mixed` rank and print its per-profile results.
- **Picking the best** (`better`, `better_mixed`): continuous measures, so one crash doesn't decide it: the distance
  completed (and in stage B first the points against the reference field), then, within 2% of it, the profile's
  reward per step. Evals before 2026-10-31 (EVAL_VERSION 1) measured other things and aren't compared.

    f1sim-eval --checkpoint runs/middle/stage-a/best.npz --tracks tracks --laps 50

v2 to beat (alone, 50 laps): finished 53%, crashes 1.39% per lap.
"""

from __future__ import annotations

import argparse
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

EVAL_VERSION = 2  # 2026-10-31: completed distance, the profile's reward per step, the reference field
DISTANCE_BAND = 0.02  # within this share of the distance completed, the reward per step decides
POINTS_BAND = 0.1  # stage B: within this many points per race, the distance and then the reward decide


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

        (state, obs), info = jax.lax.scan(body, (state, obs), jax.random.split(key, steps))
        return state, obs, info

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
        state, obs, info = run(bank, actor, state, obs, k)
        for name, v in info.items():
            if name == "amounts":
                per_term = np.asarray(v).sum(axis=0)
                totals["off_track_s"] = totals.get("off_track_s", 0.0) + float(per_term[TERMS.index("off_track")])
                if weights is not None:
                    totals["reward"] = totals.get("reward", 0.0) + float(per_term @ np.asarray(weights))
            else:
                totals[name] = totals.get(name, 0.0) + float(np.asarray(v).sum())
        steps += chunk
        if bool(np.asarray(done_fn(bank, state.race, state.limit)).all()):
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
    tyres = f"  {r['km_per_tyre_set']:.0f} km/set" if "km_per_tyre_set" in r else ""
    return f"{new}finished {100 * r['finish_rate']:.0f}%  {crashes}  lap x{r['lap_ratio']:.3f} of the lap sim  off track {100 * r.get('off_track_share', 0.0):.1f}%  {r['laps_per_stint']:.1f} laps/stint{tyres}  ({r['laps']:.0f} laps)"


def better_mixed(a: dict, b: dict | None) -> bool:
    """Stage B: more than 0.1 more points per race (against the fixed reference field), else the distance and the reward
    per step as in `better`. An older eval (another version) is no bar."""
    if b is None or _version(b) != _version(a):
        return True
    if abs(a["points_per_race"] - b["points_per_race"]) > POINTS_BAND:
        return a["points_per_race"] > b["points_per_race"]
    return _distance_then_reward(a, b)


def summary_mixed(r: dict) -> str:
    crashes = f"crashes {100 * r['crashes_per_lap']:.2f}%/lap" if r["laps"] >= 1 else f"{r['crashes_per_lap']:.0f} crashes, no lap done"
    held = f"{100 * r['attacks_held']:.0f}%" if np.isfinite(r.get("attacks_held", np.nan)) else "-"
    new = f"distance {100 * r['distance']:.1f}%  reward {r['reward_per_step']:.3f}/step  " if "distance" in r else ""
    return (
        f"{r['points_per_race']:5.2f} pts/race  {new}finished {100 * r['finish_rate']:3.0f}%  out on lap 1 {100 * r['out_lap1']:3.0f}%  "
        f"{crashes}  off track {100 * r.get('off_track_share', 0.0):.1f}%  overtakes {r['overtakes_per_race']:.2f}  lost {r['places_lost_per_race']:.2f}  attacks held {held}"
    )


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
    agent, _ = checkpoint.load(a.checkpoint, template, s.tqc)
    bank = make_bank(eval_tracks(a.tracks, a.width))
    r = alone(bank, agent.actor, s.tqc, a.laps, a.seeds, weights=s.weights)
    print(f"{info['profile']} ({a.checkpoint}): {summary(r)}")
    print("v2 middle alone, 50 laps: finished 53%, crashes 1.39%/lap")


if __name__ == "__main__":
    main()
