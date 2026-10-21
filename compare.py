"""
Head to head: two trained drivers race on full grids, half the cars each,
scored with F1 points (25-18-15-12-10-8-6-4-2-1 for the top 10, 0 for a
crash). Every grid is raced twice with the seats swapped, so neither driver
profits from luckier slots. The verdict is the F1-points difference per
race with a 95% confidence interval: a winner only when it excludes 0.

    uv run --with torch compare.py --a aggressive --b middle
    uv run --with torch compare.py --a aggressive --b middle --designs
    uv run --with torch compare.py --a aggressive:latest --b saved_models/aggressive/td3_best.pt --races 200
    uv run compare.py --leaderboard

Drivers race greedily on the normal car, every map in turn: the stock car, or
with --designs the teams of configs/designs.toml (each team's 2 cars on its
design, on random slots per grid; the seat swap keeps the cars where they
are, so each driver races every design equally). The races run in a pool of
torch-free processes; torch only loads the weights here.
"""

import argparse
import gc
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import numpy as np

from racer import config, design
from racer.console import bold, dim, duration, green, lap_time
from racer.records import LapRecords
from racer.scoring import f1_points, place_points
from racer.track import available_maps


def parse_args():
    p = argparse.ArgumentParser(description="Race two trained drivers against each other (F1 points), "
                                            "or print the lap-record leaderboard.")
    p.add_argument("--a", metavar="PROFILE[:TAG]|FILE", help="driver A, e.g. aggressive")
    p.add_argument("--b", default="middle", metavar="PROFILE[:TAG]|FILE", help="driver B (default: middle)")
    p.add_argument("--races", type=int, default=100, help="races (in pairs: every grid twice, seats swapped)")
    p.add_argument("--cars", type=int, default=22, help="cars per race, half each (even)")
    p.add_argument("--maps", default="all", help="'all', 'eval' ([eval] maps) or a list, e.g. Monza,Spa")
    p.add_argument("--seconds", type=float, default=None, help="race length (default: the training length)")
    p.add_argument("--workers", type=int, default=4, help="races run in parallel processes")
    p.add_argument("--seed", type=int, default=0, help="first grid's seed (which slots each driver gets)")
    p.add_argument("--designs", nargs="?", const="", default=None, metavar="FILE",
                   help="race the teams of the designs file (default: [design] designs_file) instead of stock cars")
    p.add_argument("--leaderboard", action="store_true", help="every profile's lap record per map, then exit")
    return p.parse_args()


def checkpoint_file(cfg, spec):
    """'aggressive', 'aggressive:latest' or a .pt path -> (path, profile)."""
    if spec.endswith(".pt") or os.sep in spec:
        path = Path(spec).resolve()
        profile = path.parent.name
        return path, profile if profile in cfg.profile_names else cfg.run.default_profile
    profile, _, tag = spec.partition(":")
    try:
        cfg.profile(profile)
    except ValueError as e:
        raise SystemExit(f"{e} - or give a .pt file") from None
    return cfg.checkpoint_path(profile, tag or "best"), profile


# ---- pool workers (torch-free) ------------------------------------------------

_CFG = None
_DRIVERS = None          # (FrozenPolicy A, FrozenPolicy B)
_ENV = None              # (map, env) - kept while the next race is on the same map


def _init(cfg, params_a, params_b):
    from racer.policy import FrozenPolicy
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _CFG, _DRIVERS
    _CFG = cfg
    _DRIVERS = (FrozenPolicy(params_a), FrozenPolicy(params_b))


def _race_pair(task):
    """One grid on one map, raced twice: A on the grid's first half of the
    slots and B on the rest, then the other way round."""
    from racer.env import RaceEnv
    global _ENV
    map_name, seed, n, steps, profiles, car_designs = task
    if _ENV is None or _ENV[0] != map_name:
        if _ENV is not None:
            _ENV[1].close()
            gc.collect()                   # the gym env has reference loops
        _ENV = (map_name, RaceEnv(_CFG, n, map_name, profiles[0], seed=seed))
    env = _ENV[1]
    env.max_steps = steps
    slots = np.random.default_rng(seed).permutation(n)
    results = []
    for a_slots in (slots[:n // 2], slots[n // 2:]):
        team = np.ones(n, dtype=np.int64)
        team[a_slots] = 0
        env.set_profiles([profiles[t] for t in team])
        env.set_designs(car_designs)
        obs = env.reset(np.random.default_rng(seed), spawn="grid")
        while not env.done.all():
            actions = np.zeros((n, _CFG.action_dim), dtype=np.float32)
            racing = ~env.done
            for t, driver in enumerate(_DRIVERS):
                idx = (team == t) & racing
                if idx.any():
                    actions[idx] = driver.act(obs[idx])
            obs, *_ = env.step(actions)
        results.append(dict(_score(env, team, map_name), grid=seed))
    return results


def _score(env, team, map_name):
    """Per driver: F1 points, place points, wins, podiums, finishers, wall /
    car crashes and passes on the other driver's cars."""
    n = env.num_agents
    out = {"map": map_name}
    for t in (0, 1):
        cars, rivals = np.flatnonzero(team == t), np.flatnonzero(team != t)
        places = sorted(env.finish_position[i] for i in cars if env.finish_position[i] is not None)
        out[t] = {"f1": sum(f1_points(p) for p in places), "place_points": sum(place_points(n, p) for p in places),
                  "places": places, "wins": int(1 in places), "podiums": sum(p <= 3 for p in places),
                  "finished": len(places), "cars": len(cars),
                  "wall": sum(env.termination_reason[i] == "collision" for i in cars),
                  "car": sum(env.termination_reason[i] == "contact" for i in cars),
                  "passes": int(env.passes_on[np.ix_(cars, rivals)].sum())}
    return out


# ---- main -----------------------------------------------------------------------

def leaderboard(cfg):
    tables = {p: LapRecords.load(cfg, p) for p in cfg.profile_names}
    tables = {p: t for p, t in tables.items() if t}
    if not tables:
        print("No lap records yet - training writes them to saved_models/<profile>/lap_records.json.")
        return
    profiles = list(tables)
    maps = sorted(available_maps(cfg))
    width = max(11, *(len(p) + 2 for p in profiles))
    print(bold("Lap records") + "\n")
    print(f"{'map':<15}" + "".join(f"{p:>{width}}" for p in profiles))
    for m in maps:
        times = {p: tables[p][m]["lap_s"] for p in profiles if m in tables[p]}
        if not times:
            continue
        fastest = min(times.values())
        cells = []
        for p in profiles:
            if p in times:
                mark = "*" if times[p] == fastest and len(times) > 1 else " "
                cells.append(f"{lap_time(times[p], unit='') + mark:>{width}}")
            else:
                cells.append(f"{'-':>{width}}")
        print(f"{m:<15}" + "".join(cells))
    print("\n" + " · ".join(f"{p}: {len(tables[p])}/{len(maps)} maps" for p in profiles)
          + ("   (* = fastest profile on that map)" if len(profiles) > 1 else ""))


def main():
    args = parse_args()
    cfg = config.load()
    if args.leaderboard:
        return leaderboard(cfg)
    if not args.a:
        raise SystemExit("give --a (and --b), or --leaderboard")
    if args.cars < 2 or args.cars % 2:
        raise SystemExit("--cars must be even (half for each driver)")
    every = available_maps(cfg)
    maps = (sorted(every) if args.maps == "all" else list(cfg.eval.maps) if args.maps == "eval"
            else [m.strip() for m in args.maps.split(",") if m.strip()])
    missing = [m for m in maps if m not in every]
    if missing:
        raise SystemExit(f"unknown map(s) {missing} - available: {', '.join(sorted(every))}")

    from racer.agent import read_actor_weights               # torch: this process only
    names, params, profiles = [], [], []
    for k, spec in enumerate((args.a, args.b)):
        path, profile = checkpoint_file(cfg, spec)
        if not path.exists():
            raise SystemExit(f"no checkpoint at '{config.shown(path)}'")
        weights, data = read_actor_weights(path, cfg)
        names.append(spec + (" (B)" if k == 1 and args.a == args.b else ""))
        params.append(weights)
        profiles.append(profile)
        print(f"{dim('·')} {names[-1]}: {config.shown(path)} " + dim(f"({data['model'].get('total_it', '?')} updates)"))

    teams = None
    if args.designs is not None:
        try:
            teams = design.read_designs(cfg, args.designs or None)
        except ValueError as e:
            raise SystemExit(str(e)) from None

    def car_designs(seed):
        """Stock cars, or every team's design on 2 random slots of this grid."""
        if teams is None:
            return None
        team_of = np.empty(args.cars, dtype=np.int64)
        team_of[np.random.default_rng([seed, 1]).permutation(args.cars)] = np.arange(args.cars) // 2 % len(teams)
        return np.stack([teams[t][1] for t in team_of])

    grids = (args.races + 1) // 2
    seconds = args.seconds if args.seconds is not None else cfg.episode_steps * cfg.sim.control_dt
    steps = int(round(seconds / cfg.sim.control_dt))
    tasks = sorted(((maps[g % len(maps)], args.seed + g, args.cars, steps, profiles, car_designs(args.seed + g))
                    for g in range(grids)), key=lambda task: task[0])        # same map in a row: fewer env rebuilds
    print(bold(f"Head to head: {names[0]} vs {names[1]}") + f" · {2 * grids} races ({grids} grids, seats swapped) · "
          f"{args.cars} cars, {args.cars // 2} each · {len(maps)} map(s) · {seconds:g} s races · {args.workers} workers")
    if teams is not None:
        print(dim("· cars: ") + ", ".join(f"{name} {design.text(d)}" for name, d in teams)
              + dim(" - 2 cars each" + (", cycled" if args.cars > 2 * len(teams) else "")))
    else:
        print(dim(f"· cars: all stock {design.text(cfg.design.stock)}"))

    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    races, t0 = [], time.time()
    pool = mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg, *params))
    try:
        for k, pair in enumerate(pool.imap_unordered(_race_pair, tasks), 1):
            races += pair
            line = " · ".join(f"{r[0]['f1']:>3} : {r[1]['f1']:<3}" for r in pair)
            print(f"  [{k:>3}/{grids}] {pair[0]['map']:<15} F1 points {names[0]} : {names[1]}   {line}"
                  + dim(f"   ({duration(time.time() - t0)})"), flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - results so far:")
    finally:
        pool.terminate()
    if races:
        report(races, names)


def report(races, names):
    n = len(races)
    keys = ("f1", "place_points", "wins", "podiums", "finished", "cars", "wall", "car", "passes")
    total = {t: {k: sum(r[t][k] for r in races) for k in keys} for t in (0, 1)}
    places = {t: [p for r in races for p in r[t]["places"]] for t in (0, 1)}
    outscored = [sum(r[0]["f1"] > r[1]["f1"] for r in races), sum(r[1]["f1"] > r[0]["f1"] for r in races)]
    rows = [("F1 points", lambda t: f"{total[t]['f1']}"),
            ("  per race", lambda t: f"{total[t]['f1'] / n:.1f}"),
            ("Race wins (P1)", lambda t: f"{total[t]['wins']}"),
            ("Podiums", lambda t: f"{total[t]['podiums']}"),
            ("Outscored the other", lambda t: f"{outscored[t]}"),
            ("Cars finished", lambda t: f"{total[t]['finished'] / total[t]['cars']:.0%}"),
            ("Crashes: wall / car", lambda t: f"{total[t]['wall']} / {total[t]['car']}"),
            ("Passes on the other", lambda t: f"{total[t]['passes']}"),
            ("Avg finishing place", lambda t: f"P{np.mean(places[t]):.1f}" if places[t] else "-"),
            ("Place points per race", lambda t: f"{total[t]['place_points'] / n:.1f}")]
    width = max(16, *(len(name) + 3 for name in names))
    print("\n" + bold(f"{n} races") + "".join(f"{name:>{width}}" for name in names))
    for label, cell in rows:
        print(f"  {label:<22}" + "".join(f"{cell(t):>{width}}" for t in (0, 1)))

    # The verdict: each grid's two races averaged (the seat swap cancels the slots' luck).
    by_grid = {}
    for r in races:
        by_grid.setdefault(r["grid"], []).append(r[0]["f1"] - r[1]["f1"])
    diffs = [float(np.mean(d)) for d in by_grid.values() if len(d) == 2]
    print()
    if len(diffs) < 2:
        print("Verdict: too few complete grids for a confidence interval - race more (--races).")
        return
    mean = float(np.mean(diffs))
    half = 1.96 * float(np.std(diffs, ddof=1)) / np.sqrt(len(diffs))
    better, worse = (names[0], names[1]) if mean > 0 else (names[1], names[0])
    ci = f"95% CI {abs(mean) - half:.1f} to {abs(mean) + half:.1f}"
    if abs(mean) > half:
        print(bold("Verdict: ") + green(f"{better} is better") + f" - {abs(mean):.1f} more F1 points per race than "
              f"{worse} ({ci}, over {len(diffs)} grids).")
    else:
        print(bold("Verdict: no clear winner") + f" - {better} scores {abs(mean):.1f} more F1 points per race, but "
              f"that's within the noise ({ci}): run more races.")


if __name__ == "__main__":
    main()
