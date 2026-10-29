"""Evals (Phase 2, section 5), on the fixed eval tracks with the default car (no upgrades), the training principal, no
respawn and each actor's mean action.

- **Alone** (`alone`): one car per race; the finish rate, crashes per lap, lap time (against each track's lap
  simulation) and tyre stints.
- **Mixed** (`mixed`): 22-car races, every seat one of the profiles (5-6 cars each) from a random grid; per profile the
  points per race, finish rate, cars out on lap 1, crashes per lap, overtakes, places lost and time held under attack.

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
from f1env.weekend import score

from .. import checkpoint, tqc
from . import env as de
from .config import PROFILES
from .rewards import TERMS


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


def mixed_seats(cars: int, profiles: int) -> np.ndarray:
    return np.arange(cars) % profiles


@lru_cache(maxsize=8)
def _mixed_runner(ecfg: de.EnvConfig, cfg: tqc.TQCConfig, steps: int, seats: tuple):
    seats = np.asarray(seats)
    idx = [np.flatnonzero(seats == p) for p in range(int(seats.max()) + 1)]

    @jax.jit
    def run(bank, actors, state, obs, acc, key):
        def body(c, k):
            state, obs, acc = c
            actions = jnp.zeros((ecfg.races, ecfg.cars, de.ACTIONS))
            for p, sl in enumerate(idx):
                a = tqc.act(actors[p], cfg, obs[:, sl].reshape(-1, de.OBS), k, mean=True)
                actions = actions.at[:, sl].set(a.reshape(ecfg.races, len(sl), de.ACTIONS))
            state, obs, out, _ = de.step(bank, state, actions, k, ecfg)
            return (state, obs, {"amounts": acc["amounts"] + out.amounts, "laps": acc["laps"] + out.lap_done}), None

        (state, obs, acc), _ = jax.lax.scan(body, (state, obs, acc), jax.random.split(key, steps))
        return state, obs, acc

    return run


def mixed(bank, actors, cfg: tqc.TQCConfig, laps: int, seeds: int = 1, key=None, chunk: int = 2000, max_steps: int = 2_000_000, names=None) -> dict:
    """22-car races on every track in the bank (`seeds` times each), `laps` laps; actors: one per profile."""
    names = names or PROFILES[: len(actors)]
    key = jax.random.PRNGKey(54321) if key is None else key
    n = int(bank.length.shape[0])
    r = n * seeds
    seats = mixed_seats(22, len(actors))
    ecfg = de.EnvConfig(races=r, cars=22, fixed_laps=laps, upgrades=False, random_grid=True, auto_reset=False, race=RaceConfig(respawn=False))
    k_reset, key = jax.random.split(key)
    state, obs = jax.jit(partial(de.reset, cfg=ecfg))(bank, k_reset, tid=jnp.tile(jnp.arange(n), seeds))
    acc = {"amounts": jnp.zeros((r, 22, len(TERMS))), "laps": jnp.zeros((r, 22))}
    run = _mixed_runner(ecfg, cfg, chunk, tuple(int(x) for x in seats))
    done_fn = jax.jit(partial(de.race_done, cfg=ecfg))
    steps = 0
    while steps < max_steps:
        key, k = jax.random.split(key)
        state, obs, acc = run(bank, tuple(actors), state, obs, acc, k)
        steps += chunk
        if bool(np.asarray(done_fn(bank, state.race, state.limit)).all()):
            break
    sc = score(state.race, bank)
    amounts, laps_done = np.asarray(acc["amounts"]), np.asarray(acc["laps"])
    line_laps = np.floor(np.asarray(state.race.line) / np.asarray(bank.length[state.race.tid])[:, None])
    out_lap1 = np.asarray(state.race.dnf) & (line_laps < 1)
    t = {name: i for i, name in enumerate(TERMS)}
    res = {"races": r, "steps": steps, "profiles": {}}
    for p, name in enumerate(names):
        sl = seats == p
        cars = sl.sum() * r
        per_car = lambda x: float(x[:, sl].sum() / cars)
        res["profiles"][name] = {
            "points_per_race": per_car(sc.points),
            "finish_rate": per_car(sc.finished),
            "out_lap1": per_car(out_lap1),
            "crashes_per_lap": float(amounts[:, sl, t["crash"]].sum() / max(laps_done[:, sl].sum(), 1.0)),
            "laps": float(laps_done[:, sl].sum()),
            "overtakes_per_race": per_car(amounts[..., t["overtake"]]),
            "places_lost_per_race": per_car(amounts[..., t["place_lost"]]),
            "held_s_per_race": per_car(amounts[..., t["holding"]]),
        }
    return res


def better_mixed(a: dict, b: dict | None) -> bool:
    """More points per race, then more finishes, then fewer crashes per lap."""
    if b is None:
        return True
    key = lambda r: (round(r["points_per_race"], 3), round(r["finish_rate"], 4), -round(r["crashes_per_lap"], 5))
    return key(a) > key(b)


def summary_mixed(r: dict) -> str:
    crashes = f"crashes {100 * r['crashes_per_lap']:.2f}%/lap" if r["laps"] >= 1 else f"{r['crashes_per_lap']:.0f} crashes, no lap done"
    return (
        f"{r['points_per_race']:5.2f} pts/race  finished {100 * r['finish_rate']:3.0f}%  out on lap 1 {100 * r['out_lap1']:3.0f}%  "
        f"{crashes}  overtakes {r['overtakes_per_race']:.2f}  lost {r['places_lost_per_race']:.2f}  held {r['held_s_per_race']:.0f} s"
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
    agent, _ = checkpoint.load(a.checkpoint, template)
    bank = make_bank(eval_tracks(a.tracks, a.width))
    r = alone(bank, agent.actor, s.tqc, a.laps, a.seeds)
    print(f"{info['profile']} ({a.checkpoint}): {summary(r)}")
    print("v2 middle alone, 50 laps: finished 53%, crashes 1.39%/lap")


if __name__ == "__main__":
    main()
