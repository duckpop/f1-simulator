"""The Phase 2 evaluation (spec section 5) and the comparison with v2.

    f1sim-season --tracks tracks                              # the full test: each profile's stage B best
    f1sim-season --rounds 2 --race-km 20 --alone-laps 2       # a quick look

1. **Alone**: each profile, 50 laps on each of the 10 eval tracks (`evaluate.alone`).
2. **Season**: 24 rounds of f1env's eval calendar (on the eval tracks, with its round keys), full weekends: Q1, Q2 and
   Q3 decide the grid, then a 305 km race and points (f1env's `F1Env` runs the weekends). Every seat is one of the
   four profiles (5-6 cars each), the default car, the training principal in the races, no respawn. Per profile:
   points per race, finish rate, out on lap 1, crashes per lap, the champion.
3. **Style**, from the season's races: overtakes per race, attacks held (a spell with a rival within 1 s behind that
   ends without losing the place), km per tyre set (from the wear rate) at the profile's race pace, time in wheelspin
   or lock-up.
4. Phase 2's done criteria checked against v2, and a report: <out>/phase2-<date>.md and .json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.env import F1Env
from f1env.weekend import EVAL_CALENDAR, WeekendConfig, eval_keys

from .. import checkpoint, tqc
from . import env as de
from . import evaluate
from .config import PROFILES, load
from .rewards import TERMS

CARS = 22
STEP_S = 0.05
V2 = {  # measured 2026-10-24/25 on v2's environment (real tracks): the bar, though the conditions differ
    "alone": {"finish_rate": 0.53, "crashes_per_lap": 0.0139},
    "season": {"middle": {"finish_rate": 0.36, "out_lap1": 0.28}, "aggressive": {"finish_rate": 0.34, "out_lap1": 0.29}, "defensive": {"finish_rate": 0.32, "out_lap1": 0.30}},
}
SIMILAR_PACE = 0.02  # Smooth's tyre life counts at a race speed within this of the profiles' median
WITHIN_NOISE = 0.05  # Middle's points count as the most within this of the best


class Attacks(NamedTuple):
    on: jax.Array  # (R, C) in a spell under attack
    count: jax.Array  # spells
    lost: jax.Array  # spells that ended with the place lost


def track_attacks(a: Attacks, attack, lost) -> Attacks:
    lose = (lost > 0) & (a.on | attack)
    return Attacks(attack & ~lose, a.count + (attack & ~a.on), a.lost + lose)


class Tally(NamedTuple):
    """Per car, over a session."""

    amounts: jax.Array  # (R, C, terms)
    laps: jax.Array
    valid: jax.Array  # steps driven
    slipping: jax.Array  # steps in wheelspin or a lock-up
    attacks: Attacks


def new_tally(r: int, c: int) -> Tally:
    z = jnp.zeros((r, c))
    return Tally(jnp.zeros((r, c, len(TERMS))), z, z, z, Attacks(z > 0, z, z))


def seats_for(profiles: int = len(PROFILES), cars: int = CARS) -> np.ndarray:
    """Car k drives profile k mod 4: 6, 6, 5, 5 cars."""
    return np.arange(cars) % profiles


@lru_cache(maxsize=4)
def _runner(ecfg: de.EnvConfig, cfg: tqc.TQCConfig, steps: int, seats: tuple):
    seats = np.asarray(seats)
    idx = [np.flatnonzero(seats == p) for p in range(int(seats.max()) + 1)]

    @jax.jit
    def run(bank, actors, state, obs, tally, key):
        def body(c, k):
            state, obs, t = c
            actions = jnp.zeros((ecfg.races, ecfg.cars, de.ACTIONS))
            for p, sl in enumerate(idx):
                a = tqc.act(actors[p], cfg, obs[:, sl].reshape(-1, de.OBS), k, mean=True)
                actions = actions.at[:, sl].set(a.reshape(ecfg.races, len(sl), de.ACTIONS))
            state, obs, out, _ = de.step(bank, state, actions, k, ecfg)
            t = Tally(t.amounts + out.amounts, t.laps + out.lap_done, t.valid + out.valid, t.slipping + out.slipping, track_attacks(t.attacks, out.attack, out.lost))
            return (state, obs, t), None

        (state, obs, tally), _ = jax.lax.scan(body, (state, obs, tally), jax.random.split(key, steps))
        return state, obs, tally

    return run


def run_session(env: F1Env, session, actors, cfg: tqc.TQCConfig, seats, ecfg: de.EnvConfig, key, chunk: int = 2000, max_steps: int = 400_000):
    """Drive an F1Env session to its end with the actors (mean actions). Returns (session, Tally, steps)."""
    k_wrap, key = jax.random.split(key)
    state, obs = de.compiled("from_race", ecfg)(env.bank, session.state, k_wrap, plan=None if session.race else session.plan)
    tally = new_tally(*session.state.s.shape)
    run = _runner(ecfg, cfg, chunk, tuple(int(x) for x in seats))
    done_fn = de.compiled("race_done", ecfg)
    steps = 0
    while steps < max_steps:
        key, k = jax.random.split(key)
        state, obs, tally = run(env.bank, tuple(actors), state, obs, tally, k)
        steps += chunk
        if bool(np.asarray(done_fn(env.bank, state.race, state.limit)).all()):
            break
    session.state = state.race
    return session, tally, steps


def season(tracks, actors, cfg: tqc.TQCConfig, seasons: int = 1, rounds: int = 24, race_km: float = 305.0, chunk: int = 2000, env: F1Env | None = None, log=print) -> dict:
    """Full weekends over f1env's eval calendar; the four profiles' actors in seats_for()."""
    seats = seats_for(len(actors))
    env = env or F1Env(tracks, WeekendConfig(race_km=race_km))
    wcfg = env.wcfg
    cal = np.tile(np.asarray(EVAL_CALENDAR[:rounds]), (seasons, 1))
    s = env.reset_season(jax.random.PRNGKey(2026), calendar=cal, profiles=np.tile(seats, (seasons, 1)))
    ecfg = de.EnvConfig(races=seasons, cars=CARS, upgrades=False, auto_reset=False, quali=True, race=wcfg.race)
    keys = eval_keys()
    total, out_lap1, grids = None, [], []
    for rnd in range(rounds):
        t0, n = time.time(), 0
        for i, kind in enumerate(("Q1", "Q2", "Q3", "race")):
            k = jax.random.fold_in(keys[rnd], i)
            session = env.start_session(s, kind, key=k)
            session, tally, steps = run_session(env, session, actors, cfg, seats, ecfg, k, chunk)
            n += steps
            if kind == "race":
                total = tally if total is None else jax.tree.map(jnp.add, total, tally)
                st = session.state
                laps = np.floor(np.asarray(st.line) / np.asarray(env.bank.length[st.tid])[:, None])
                out_lap1.append(np.asarray(st.dnf) & (laps < 1))
                grids.append(np.asarray(s.quali["grid"]))
            s = env.end_session(s, session)
        h = s.history[-1]
        winner = np.argmin(h["place"], axis=-1)
        if log:
            log(
                f"round {rnd + 1:>2}/{rounds} ({env.tracks[cal[0, rnd]].name}): won by {', '.join(PROFILES[seats[w]] for w in winner)}; "
                f"finished {100 * np.mean(h['finished']):.0f}%  ({n:,} steps in {time.time() - t0:.0f} s)",
                flush=True,
            )
    return summarise(s, total, np.stack(out_lap1, axis=1), np.stack(grids, axis=1), seats, rounds)


def summarise(s, total: Tally, out_lap1, grids, seats, rounds: int) -> dict:
    """Per profile, from the season's history (points, finishes, speed) and the races' tallies (out_lap1: (R, rounds,
    C)). grids: (R, rounds, C) each car's grid slot."""
    hist = {k: np.stack([h[k] for h in s.history], axis=1) for k in ("points", "finished", "speed", "dsq")}  # (R, rounds, C)
    t = {name: i for i, name in enumerate(TERMS)}
    amounts = np.asarray(total.amounts)
    laps, valid, slipping = (np.asarray(x) for x in (total.laps, total.valid, total.slipping))
    att_count, att_lost = np.asarray(total.attacks.count), np.asarray(total.attacks.lost)
    champions = seats[s.standings.drivers()[:, 0]]
    res = {"seasons": int(hist["points"].shape[0]), "rounds": rounds, "profiles": {}, "out_lap1_field": float(out_lap1.mean()), "champions": [PROFILES[c] for c in champions]}
    for p, name in enumerate(PROFILES[: int(seats.max()) + 1]):
        sl = seats == p
        cars = int(sl.sum()) * hist["points"].shape[0]
        per_race = lambda x: float(x[..., sl].sum() / (cars * rounds))
        fin = hist["finished"][..., sl]
        speed = hist["speed"][..., sl]
        km = amounts[:, sl, t["progress"]].sum() / 1000.0
        wear = amounts[:, sl, t["tyre_wear"]].sum() / 100.0  # tyre sets' worth of life used
        res["profiles"][name] = {
            "points_per_race": per_race(hist["points"]),
            "finish_rate": float(fin.mean()),
            "out_lap1": float(out_lap1[..., sl].mean()),
            "disqualified": float(hist["dsq"][..., sl].mean()),
            "crashes_per_lap": float(amounts[:, sl, t["crash"]].sum() / max(laps[:, sl].sum(), 1.0)),
            "laps": float(laps[:, sl].sum()),
            "race_speed_kmh": float(speed[fin].mean()) if fin.any() else float("nan"),
            "overtakes_per_race": float(amounts[:, sl, t["overtake"]].sum() / (cars * rounds)),
            "places_lost_per_race": float(amounts[:, sl, t["place_lost"]].sum() / (cars * rounds)),
            "attacks_per_race": float(att_count[:, sl].sum() / (cars * rounds)),
            "attacks_held": float(1.0 - att_lost[:, sl].sum() / att_count[:, sl].sum()) if att_count[:, sl].sum() else float("nan"),
            "km_per_tyre_set": float(km / wear) if wear > 0 else float("nan"),
            "slip_s_per_race": float(slipping[:, sl].sum() * STEP_S / (cars * rounds)),
            "slip_share": float(slipping[:, sl].sum() / max(valid[:, sl].sum(), 1.0)),
            "mean_grid_slot": float(grids[..., sl].mean() + 1),
            "titles": int(sum(c == p for c in champions)),
        }
    return res


def _leads(res: dict, metric: str, who: str) -> bool:
    """`who` is ahead of every other profile (a tie isn't a lead)."""
    vals = {p: r[metric] for p, r in res.items() if np.isfinite(r[metric])}
    return who in vals and all(vals[who] > v for p, v in vals.items() if p != who)


def _pct(x: float, laps: float | None = None, digits: int = 0) -> str:
    if laps is not None and laps < 1:
        return "no lap done"
    return "-" if not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _num(x: float, fmt: str = ".0f", unit: str = "") -> str:
    return "-" if not np.isfinite(x) else f"{x:{fmt}}{unit}"


def criteria(alone: dict | None, sea: dict | None) -> list[tuple[str, bool | None, str]]:
    """Phase 2's done criteria: (what, passed (None: not measured), the numbers)."""
    out = []
    v2a = V2["alone"]
    for name in PROFILES:
        if alone and name in alone:
            r = alone[name]
            ok = r["finish_rate"] > v2a["finish_rate"] and r["laps"] >= 1 and r["crashes_per_lap"] < v2a["crashes_per_lap"]
            out.append((f"{name} beats v2 Middle alone", ok, f"finished {_pct(r['finish_rate'])} (v2 53%), crashes per lap {_pct(r['crashes_per_lap'], r['laps'], 2)} (v2 1.39%)"))
        else:
            out.append((f"{name} beats v2 Middle alone", None, "not run"))
    if sea:
        f = sea["out_lap1_field"]
        out.append(("fewer cars out on lap 1 than v2 (28%)", f < V2["season"]["middle"]["out_lap1"], f"{100 * f:.0f}% of cars"))
        pr = sea["profiles"]
        out.append(("Aggressive makes the most overtakes", _leads(pr, "overtakes_per_race", "aggressive"), ", ".join(f"{p} {r['overtakes_per_race']:.2f}" for p, r in pr.items())))
        out.append(("Defensive holds the most attacks", _leads(pr, "attacks_held", "defensive"), ", ".join(f"{p} {_pct(r['attacks_held'])} of {r['attacks_per_race']:.1f}/race" for p, r in pr.items())))
        speeds = [r["race_speed_kmh"] for r in pr.values() if np.isfinite(r["race_speed_kmh"])]
        sm = pr.get("smooth", {})
        similar = bool(speeds) and np.isfinite(sm.get("race_speed_kmh", np.nan)) and sm["race_speed_kmh"] >= (1 - SIMILAR_PACE) * float(np.median(speeds))
        out.append(("Smooth's tyres last longest at a similar pace", _leads(pr, "km_per_tyre_set", "smooth") and similar, ", ".join(f"{p} {_num(r['km_per_tyre_set'], unit=' km')} at {_num(r['race_speed_kmh'], unit=' km/h')}" for p, r in pr.items())))
        best = max(r["points_per_race"] for r in pr.values())
        mid = pr.get("middle", {}).get("points_per_race", 0.0)
        out.append(("Middle scores the most points (or within the noise)", best > 0 and mid >= (1 - WITHIN_NOISE) * best, ", ".join(f"{p} {r['points_per_race']:.2f}" for p, r in pr.items())))
    else:
        out.append(("the season's criteria", None, "not run"))
    return out


def report(alone: dict | None, sea: dict | None, sources: dict, settings: dict) -> str:
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Phase 2 results ({when})", "", "Drivers: " + ", ".join(f"{p} `{src}`" for p, src in sources.items()), ""]
    lines += [f"Settings: {', '.join(f'{k} {v}' for k, v in settings.items())}", ""]
    lines += ["## Done criteria", "", "| | Criterion | Numbers |", "|---|---|---|"]
    for what, ok, detail in criteria(alone, sea):
        mark = "-" if ok is None else ("✓" if ok else "✗")
        lines.append(f"| {mark} | {what} | {detail} |")
    if alone:
        lines += ["", "## Alone", "", "| Profile | Finished | Crashes per lap | Lap time vs the lap simulation | Laps per stint |", "|---|---|---|---|---|"]
        for p, r in alone.items():
            ratio = f"x{r['lap_ratio']:.3f}" if r["lap_ratio"] > 0 else "-"
            lines.append(f"| {p} | {_pct(r['finish_rate'])} | {_pct(r['crashes_per_lap'], r['laps'], 2)} | {ratio} | {r['laps_per_stint']:.1f} |")
        lines.append("| v2 Middle | 53% | 1.39% | | |")
    if sea:
        lines += ["", f"## Season ({sea['seasons']} x {sea['rounds']} rounds, full weekends)", ""]
        lines += ["| Profile | Points per race | Finished | Out on lap 1 | Crashes per lap | Disqualified | Average grid slot | Titles |", "|---|---|---|---|---|---|---|---|"]
        for p, r in sea["profiles"].items():
            lines.append(f"| {p} | {r['points_per_race']:.2f} | {_pct(r['finish_rate'])} | {_pct(r['out_lap1'])} | {_pct(r['crashes_per_lap'], r['laps'], 2)} | {_pct(r['disqualified'])} | {r['mean_grid_slot']:.1f} | {r['titles']} |")
        for p, v in V2["season"].items():
            lines.append(f"| v2 {p} | | {100 * v['finish_rate']:.0f}% | {100 * v['out_lap1']:.0f}% | | | | |")
        lines += ["", "## Style", "", "| Profile | Overtakes per race | Places lost per race | Attacks per race | Attacks held | km per tyre set | Race speed | Wheelspin / lock-up per race |", "|---|---|---|---|---|---|---|---|"]
        for p, r in sea["profiles"].items():
            lines.append(
                f"| {p} | {r['overtakes_per_race']:.2f} | {r['places_lost_per_race']:.2f} | {r['attacks_per_race']:.1f} | {_pct(r['attacks_held'])} | "
                f"{_num(r['km_per_tyre_set'])} | {_num(r['race_speed_kmh'], unit=' km/h')} | {r['slip_s_per_race']:.1f} s ({_pct(r['slip_share'], digits=1)}) |"
            )
    lines += ["", "v2 was measured on real tracks; v3 uses generated tracks with run-offs and pits, so the v2 numbers are the bar but the conditions differ."]
    return "\n".join(lines) + "\n"


def load_actors(root, stage: str = "b"):
    """Each profile's best actor (stage B, else stage A with stage "b"); (actors, cfg, sources)."""
    actors, sources, cfg = [], {}, None
    for p in PROFILES:
        stages = ["stage-b", "stage-a"] if stage == "b" else ["stage-a"]
        path = next((Path(root) / p / st / "best.npz" for st in stages if (Path(root) / p / st / "best.npz").exists()), None)
        if path is None:
            raise FileNotFoundError(f"no {' or '.join(stages)} best.npz for {p} under {root}")
        info = checkpoint.meta(path)
        s = load(p, info.get("preset", "laptop"))
        agent, _ = checkpoint.load(path, tqc.init(jax.random.PRNGKey(0), s.tqc))
        actors.append(agent.actor)
        sources[p] = str(path)
        cfg = cfg or s.tqc
    return actors, cfg, sources


def main(argv=None):
    from .tracks import eval_tracks

    ap = argparse.ArgumentParser(description="Phase 2's evaluation: alone, a season of full weekends, style; compared with v2.")
    ap.add_argument("--checkpoints", default="runs", help="the training output directory (runs/<profile>/stage-*/best.npz)")
    ap.add_argument("--stage", default="b", choices=["a", "b"], help="which best checkpoints (b falls back to a)")
    ap.add_argument("--tracks", help="track pool directory; default: generate the eval tracks")
    ap.add_argument("--alone-laps", type=int, default=50, help="0 to skip the alone test")
    ap.add_argument("--rounds", type=int, default=24, help="0 to skip the season")
    ap.add_argument("--seasons", type=int, default=1, help="parallel seasons (different weather, lights, stops)")
    ap.add_argument("--race-km", type=float, default=305.0)
    ap.add_argument("--out", default="runs/results")
    a = ap.parse_args(argv)

    actors, cfg, sources = load_actors(a.checkpoints, a.stage)
    tracks = eval_tracks(a.tracks)
    env = F1Env(tracks, WeekendConfig(race_km=a.race_km))
    alone = None
    if a.alone_laps:
        alone = {}
        for p, actor in zip(PROFILES, actors):
            alone[p] = evaluate.alone(env.bank, actor, cfg, a.alone_laps)
            print(f"alone {p:>10}: {evaluate.summary(alone[p])}", flush=True)
    sea = season(tracks, actors, cfg, a.seasons, a.rounds, a.race_km, env=env) if a.rounds else None
    settings = {"alone laps": a.alone_laps, "rounds": a.rounds, "seasons": a.seasons, "race km": a.race_km, "eval tracks": len(tracks)}
    text = report(alone, sea, sources, settings)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"phase2-{dt.datetime.now():%Y%m%d-%H%M}"
    stem.with_suffix(".md").write_text(text)
    stem.with_suffix(".json").write_text(json.dumps({"alone": alone, "season": sea, "sources": sources, "settings": settings, "criteria": criteria(alone, sea)}, indent=1))
    print(text)
    print(f"written: {stem}.md, {stem}.json")


if __name__ == "__main__":
    main()
