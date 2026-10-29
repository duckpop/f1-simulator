"""Watch a trained driver: one race recorded with f1env's viewer, written as a single HTML page to open in any browser.

    f1sim-watch --tracks tracks                          # Middle's current best, alone, on eval track 0, 5 laps
    f1sim-watch --tracks tracks --cars 22 --track 3      # the same driver in all 22 cars
    f1sim-watch --checkpoint runs/middle/stage-a/best-26.npz --tracks tracks

The default checkpoint is the profile's newest best: stage B's best.npz, else stage A's best.npz, else the newest
best-<width>.npz (the curriculum's current width), else latest.npz. It races on the eval track at the width the
checkpoint was saved at (the normal one if that track has no copy at that width), with its mean action, the default
car (no upgrades) and the training principal calling the stops; no respawn unless --respawn.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.race import RaceConfig
from f1env.sensors import make_bank
from f1env.viewer import Recorder, write_html

from .. import checkpoint, tqc
from . import env as de
from .config import PROFILES, load
from .tracks import eval_track, widened


def newest_best(runs, profile: str) -> Path:
    base = Path(runs) / profile
    for p in (base / "stage-b" / "best.npz", base / "stage-a" / "best.npz"):
        if p.exists():
            return p
    widths = sorted((base / "stage-a").glob("best-*.npz"), key=lambda p: p.stat().st_mtime)
    if widths:
        return widths[-1]
    if (base / "stage-a" / "latest.npz").exists():
        return base / "stage-a" / "latest.npz"
    raise FileNotFoundError(f"no checkpoint for {profile} under {base} yet (the first is saved at the first eval)")


def record(track, actor, cfg: tqc.TQCConfig, cars: int = 1, laps: int = 5, respawn: bool = False, seconds: float = 0.0, seed: int = 0, title: str = "", max_steps: int = 400_000):
    """(Recorder, final RaceState) for one race on `track` with `actor` in every car."""
    bank = make_bank([track])
    ecfg = de.EnvConfig(races=1, cars=cars, fixed_laps=laps, upgrades=False, random_grid=cars > 1, auto_reset=False, race=RaceConfig(respawn=respawn))

    @jax.jit
    def one(bank, actor, st, obs, k):
        a = tqc.act(actor, cfg, obs.reshape(-1, de.OBS), k, mean=True).reshape(1, cars, de.ACTIONS)
        st, obs, _, _, ev = de.step(bank, st, a, k, ecfg, events=True)
        return st, obs, ev

    key = jax.random.PRNGKey(seed)
    k_reset, key = jax.random.split(key)
    state, obs = de.compiled("reset", ecfg)(bank, k_reset, tid=jnp.zeros(1, jnp.int32))
    done = de.compiled("race_done", ecfg)
    rec = Recorder(track, race=0, title=title or track.name)
    for n in range(1, max_steps + 1):
        key, k = jax.random.split(key)
        state, obs, ev = one(bank, actor, state, obs, k)
        race, events = jax.device_get((state.race, ev))
        rec.add(race, events)
        if (seconds and race.clock[0] >= seconds) or (n % 20 == 0 and bool(np.asarray(done(bank, state.race, state.limit))[0])):
            break
    return rec, jax.device_get(state.race)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Record a trained driver's race and write it as an HTML page.")
    ap.add_argument("--profile", default="middle", choices=PROFILES)
    ap.add_argument("--checkpoint", help="default: the profile's newest best under --runs")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tracks", help="track pool directory; default: generate the eval track")
    ap.add_argument("--track", type=int, default=0, help="eval track 0-9")
    ap.add_argument("--width", help='"normal", 26 or 20 (default: the width the checkpoint was saved at)')
    ap.add_argument("--cars", type=int, default=1, help="1 (alone) up to 22, all driven by this driver")
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this much race time (0: the whole race)")
    ap.add_argument("--respawn", action="store_true", help="crashed cars rejoin from the pit exit (as in training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="default: runs/watch/<profile>-<track>-<width>.html")
    a = ap.parse_args(argv)

    path = Path(a.checkpoint) if a.checkpoint else newest_best(a.runs, a.profile)
    info = checkpoint.meta(path)
    s = load(info.get("profile", a.profile), info.get("preset", "laptop"))
    agent, _ = checkpoint.load(path, tqc.init(jax.random.PRNGKey(0), s.tqc))
    width = a.width or info.get("width", "normal")
    base = eval_track(a.tracks, a.track)
    track = next(iter(widened([base], width)), None)
    if track is None:
        print(f"{base.name} has no copy at {width} m: racing on the normal track")
        track, width = base, "normal"
    title = f"{s.profile} ({info.get('env_steps', 0):,} steps, stage {info.get('stage', '?')}) on {base.name}, width {width}"
    print(f"recording: {title} from {path}", flush=True)
    rec, race = record(track, agent.actor, s.tqc, a.cars, a.laps, a.respawn, a.seconds, a.seed, title)

    laps = np.floor(race.line[0] / track.length).astype(int)
    finished, out = race.finish_pos[0] > 0, race.dnf[0]
    best = np.where(np.isfinite(race.best_lap[0]), race.best_lap[0], np.nan)
    print(f"race time {race.clock[0]:.0f} s: {int(finished.sum())} of {a.cars} finished, {int(out.sum())} crashed; most laps {max(int(laps.max()), 0)} of {a.laps}")
    if np.isfinite(best).any():
        print(f"best lap {np.nanmin(best):.1f} s (the lap simulation's ideal: {track.lap['lap_time_s']:.1f} s)")
    dest = Path(a.out or Path(a.runs) / "watch" / f"{s.profile}-{base.name}-{width}.html")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_html(dest, rec.replay())
    print(f"written: {dest}  (open it in a browser: xdg-open {dest})")


if __name__ == "__main__":
    main()
