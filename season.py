"""
One season with the trained designer: [season] teams x 2 cars, every team a
uniformly random legal car and two random drivers from the [season]
profiles, every track once in a random order; before each race the designer
adds or moves up to max_moves points on every team's car. Prints the
championship (drivers and teams) and each team's car evolution.

    uv run --with torch season.py --seed 3
    uv run --with torch season.py --seed 3 --save-designs     # the final cars -> configs/designs.toml

Races run greedily in this process (~13 s each); torch only loads the weights.
"""

import argparse
import os
import time

import numpy as np

from racer import config, design
from racer.console import bold, dim, duration, green, hex_color
from racer.policy import FrozenPolicy, greedy_ops
from racer.season import EnvCache, Season, play
from racer.track import available_maps


def parse_args(cfg):
    p = argparse.ArgumentParser(description="Race one season with the trained designer.")
    p.add_argument("--seed", type=int, default=0, help="the season: starting cars, drivers, calendar and grids")
    p.add_argument("--tag", default="best", help="designer checkpoint: best, latest or an update count")
    p.add_argument("--drivers", default="", metavar="PROFILE[:TAG],...",
                   help=f"the drivers teams draw from (default: {', '.join(cfg.season.profiles)} at best)")
    p.add_argument("--races", type=int, default=cfg.season.races, help="races in the season")
    p.add_argument("--save-designs", action="store_true",
                   help=f"write the final {cfg.season.teams} cars to {cfg.design.designs_file}")
    return p.parse_args()


def evolution(season, t):
    """'4-3-2-3-5 · R2 4-4-2-3-4 · R5 ...': the start car, then each new car
    and the race it was first raced in."""
    parts, last = [design.text(season.start[t])], season.start[t]
    for r, h in enumerate(season.history):
        if not np.array_equal(h["designs"][t], last):
            last = h["designs"][t]
            parts.append(f"R{r + 1} {design.text(last)}")
    return " · ".join(parts)


def main():
    cfg = config.load()
    args = parse_args(cfg)
    if args.races < 1:
        raise SystemExit("--races must be >= 1")
    cfg = cfg.override({"season.races": args.races}) if args.races != cfg.season.races else cfg

    from racer.agent import read_designer_weights, read_season_drivers
    path = cfg.designer_path(args.tag)
    if not path.exists():
        raise SystemExit(f"no designer at '{config.shown(path)}' - train it first (train_designer.py)")
    params, data = read_designer_weights(path, cfg)
    drivers, driver_ids = read_season_drivers(cfg, args.drivers)
    meta = data.get("meta", {})
    print(f"{dim('·')} designer {config.shown(path)} " + dim(
        f"({data['model'].get('total_it', '?')} updates"
        + (f", test score {meta['score']:.2f} pts/race" if "score" in meta else "") + ")"))
    print(f"{dim('·')} drivers {', '.join(driver_ids)}")

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    policies = {p: FrozenPolicy(w) for p, w in drivers.items()}
    profiles = [p for p in cfg.season.profiles if p in policies]
    season = Season(cfg, np.random.default_rng([args.seed]), available_maps(cfg), profiles)
    T = season.teams
    names = [f"Team {t + 1}" for t in range(T)]
    print(bold(f"Season {args.seed}") + f" · {T} teams × 2 cars · {len(season.calendar)} races: "
          + ", ".join(season.calendar))
    envs = EnvCache(cfg, 2 * T, args.seed)
    t0 = time.time()

    def on_race(taken, points):
        h = season.history[-1]
        changed = int((h["designs"] != (season.history[-2]["designs"] if len(season.history) > 1
                                        else season.start)).any(axis=1).sum())
        won = np.argwhere(h["places"] == 1)
        winner = f"{names[won[0][0]]} #{won[0][1] + 1} ({season.drivers[won[0][0], won[0][1]]})" if len(won) else "-"
        top = np.argsort(-points)[:3]
        print(f"  R{season.race:>2} {h['map']:<15} {changed:>2} car(s) changed · winner {winner:<30} · "
              + ", ".join(f"{names[t]} {points[t]:g}" for t in top)
              + dim(f"   ({duration(time.time() - t0)})"), flush=True)

    try:
        play(cfg, season, policies, envs, lambda obs, mask: greedy_ops(params, obs, mask),
             lambda r: np.random.default_rng([args.seed, r]), on_race=on_race)
    except KeyboardInterrupt:
        print("\ninterrupted - the season so far:")
    finally:
        envs.close()
    if not season.race:
        return

    places = np.stack([h["places"] for h in season.history])           # (races, T, 2), 0 = out
    print("\n" + bold(f"Drivers' championship after {season.race} races"))
    print(f"  {'pos':>4} {'driver':<12} {'profile':<11} {'points':>6} {'wins':>5} {'podiums':>7} {'finished':>9}")
    order = sorted(((t, d) for t in range(T) for d in range(2)), key=lambda td: -season.points[td])
    for k, (t, d) in enumerate(order, 1):
        p = places[:, t, d]
        prof = season.drivers[t, d]
        print(f"  {k:>4} {names[t] + f' #{d + 1}':<12} {hex_color(f'{prof:<11}', cfg.color(prof))} "
              f"{season.points[t, d]:>6g} {int((p == 1).sum()):>5} {int(((p >= 1) & (p <= 3)).sum()):>7} "
              f"{int((p > 0).sum()):>5}/{season.race}")

    print("\n" + bold("Teams' championship"))
    print(f"  {'pos':>4} {'team':<8} {'drivers':<24} {'points':>6}  {'start car':<10} {'final car':<10} "
          f"{'pts used':>8}")
    team_points = season.points.sum(axis=1)
    for k, t in enumerate(np.argsort(-team_points, kind="stable"), 1):
        print(f"  {k:>4} {names[t]:<8} {' + '.join(season.drivers[t]):<24} {team_points[t]:>6g}  "
              f"{design.text(season.start[t]):<10} {design.text(season.designs[t]):<10} "
              f"{int(season.designs[t].sum()):>8}")

    print("\n" + bold("Car evolution") + dim(f" ({', '.join(cfg.design.categories)}; R = from that race on)"))
    for t in range(T):
        print(f"  {names[t]:<8} {evolution(season, t)}")

    if args.save_designs:
        written = design.write_designs(cfg, [(names[t], season.designs[t]) for t in range(T)],
                                       note=f"season.py --seed {args.seed}: the cars after {season.race} races")
        print(f"\n{green('✓')} wrote the final cars to {config.shown(written)}")


if __name__ == "__main__":
    main()
