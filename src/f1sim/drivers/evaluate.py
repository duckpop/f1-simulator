"""Evals (Phase 2, section 5). So far the "alone" test: one car per race on the fixed eval tracks, the default car (no
upgrades), the training principal, no respawn, the actor's mean action. It measures the finish rate, crashes per lap,
lap time (against each track's lap simulation) and tyre stints.

    f1sim-eval --checkpoint runs/middle/stage-a/best.npz --tracks tracks --laps 50

v2 to beat (alone, 50 laps): finished 53%, crashes 1.39% per lap.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache, partial

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.race import RaceConfig

from .. import checkpoint, tqc
from . import env as de


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


def alone(bank, actor, cfg: tqc.TQCConfig, laps: int, seeds: int = 1, key=None, chunk: int = 2000, max_steps: int = 2_000_000) -> dict:
    """Every track in the bank `seeds` times, `laps` laps each."""
    key = jax.random.PRNGKey(12345) if key is None else key
    n = int(bank.length.shape[0])
    tid = jnp.tile(jnp.arange(n), seeds)
    ecfg = eval_config(n * seeds, laps)
    k_reset, key = jax.random.split(key)
    state, obs = jax.jit(partial(de.reset, cfg=ecfg))(bank, k_reset, tid=tid)
    run = _runner(ecfg, cfg, chunk)
    done_fn = jax.jit(partial(de.race_done, cfg=ecfg))
    totals: dict[str, float] = {}
    steps = 0
    while steps < max_steps:
        key, k = jax.random.split(key)
        state, obs, info = run(bank, actor, state, obs, k)
        for name, v in info.items():
            if name != "amounts":
                totals[name] = totals.get(name, 0.0) + float(np.asarray(v).sum())
        steps += chunk
        if bool(np.asarray(done_fn(bank, state.race, state.limit)).all()):
            break
    laps_done = totals.get("laps", 0.0)
    stops = totals.get("pit_stops", 0.0)
    return {
        "races": n * seeds,
        "laps": laps_done,
        "finish_rate": totals.get("finishes", 0.0) / (n * seeds),
        "crashes_per_lap": totals.get("crashes", 0.0) / max(laps_done, 1.0),
        "lap_ratio": totals.get("lap_ratio", 0.0) / max(totals.get("laps_counted", 0.0), 1.0),
        "laps_per_stint": laps_done / max(n * seeds + stops, 1.0),
        "dead_tyre_s": totals.get("dead_tyre", 0.0) * 0.05,
        "steps": steps,
    }


def better(a: dict, b: dict | None) -> bool:
    """Is eval a better than b: a higher finish rate, then fewer crashes per lap, then faster laps."""
    if b is None:
        return True
    key = lambda r: (round(r["finish_rate"], 4), -round(r["crashes_per_lap"], 5), -r["lap_ratio"] if r["lap_ratio"] > 0 else -9.0)
    return key(a) > key(b)


def summary(r: dict) -> str:
    crashes = f"crashes {100 * r['crashes_per_lap']:.2f}%/lap" if r["laps"] >= 1 else f"{r['crashes_per_lap']:.0f} crashes, no lap done"
    return f"finished {100 * r['finish_rate']:.0f}%  {crashes}  lap x{r['lap_ratio']:.3f} of the lap sim  {r['laps_per_stint']:.1f} laps/stint  ({r['laps']:.0f} laps)"


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
    r = alone(bank, agent.actor, s.tqc, a.laps, a.seeds)
    print(f"{info['profile']} ({a.checkpoint}): {summary(r)}")
    print("v2 middle alone, 50 laps: finished 53%, crashes 1.39%/lap")


if __name__ == "__main__":
    main()
