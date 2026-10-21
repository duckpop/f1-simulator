"""
Progress plots from a profile's metrics.csv and evals.csv, saved as
progress.png in its checkpoint folder (x axis: gradient updates).

    uv run --with matplotlib report.py --profile middle
"""

import argparse
from collections import defaultdict

from racer import config
from racer.records import read_csv


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if r.get(key) not in (None, ""):
            xs.append(float(r["updates"]))
            ys.append(float(r[key]))
    return xs, ys


def main():
    cfg = config.load()
    p = argparse.ArgumentParser(description="Plot a profile's training progress.")
    p.add_argument("--profile", default=cfg.run.default_profile, choices=cfg.profile_names)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = cfg.profile_dir(args.profile)
    metrics, evals = read_csv(folder / "metrics.csv"), read_csv(folder / "evals.csv")
    if not metrics and not evals:
        raise SystemExit(f"no metrics.csv / evals.csv in {config.shown(folder)}/ yet - train first")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Training progress · profile '{args.profile}' (x axis: gradient updates)")

    a = ax[0, 0]
    for key, label in (("median_distance_m", "median"), ("mean_distance_m", "mean")):
        a.plot(*series(metrics, key), label=label)
    a.set_title("Training distance per car-episode (m)")
    a.legend()

    a = ax[0, 1]
    by_map = defaultdict(list)
    for r in evals:
        by_map[r["map"]].append(r)
    for m, rows in sorted(by_map.items()):
        a.plot(*series(rows, "mean_distance_m"), marker=".", lw=1, label=m)
    rounds = list({r["updates"]: r for r in evals}.values())          # one point per round
    a.plot(*series(rounds, "round_distance_m"), color="black", lw=2, marker="o", label="round average")
    a.set_title("Eval mean distance per map (m)")
    a.legend(fontsize=7, loc="lower left")
    xs, ys = series(rounds, "round_points")
    if xs:
        a2 = a.twinx()
        a2.plot(xs, ys, color="purple", lw=1.5, ls="--", marker="x", label="race points per car")
        a2.set_ylabel("points per car")
        a2.legend(fontsize=7, loc="lower right")

    a = ax[0, 2]
    for key, label in (("survival_rate", "survived"), ("crash_rate", "crashed"), ("car_crash_rate", "into a car"),
                       ("stuck_rate", "stuck"), ("lap_rate", "≥1 lap")):
        a.plot(*series(metrics, key), label=label)
    a.set_ylim(0, 1)
    a.set_title("Endings (share of training car-episodes)")
    a.legend()

    a = ax[1, 0]
    for key, label in (("avg_speed_ms", "average"), ("top_speed_ms", "top")):
        a.plot(*series(metrics, key), label=label)
    a.set_title("Speed (m/s)")
    a.legend()

    a = ax[1, 1]
    a.plot(*series(metrics, "passes_per_car"), label="passes per car")
    a.plot(*series(metrics, "close_calls_per_car"), label="close-call steps per car")
    a2 = a.twinx()
    a2.plot(*series(metrics, "tow_frac"), color="gray", ls="--", label="time in a tow")
    xs, ys = series(metrics, "win_rate_vs_opponents")
    if xs:
        a2.plot(xs, ys, color="purple", ls=":", label="ahead of the opponents")
    a2.set_ylim(0, 1)
    a.set_title("Racing")
    a.legend(loc="upper left", fontsize=8)
    a2.legend(loc="upper right", fontsize=8)

    a = ax[1, 2]
    a.plot(*series(metrics, "avg_q"), label="avg Q")
    a.plot(*series(metrics, "critic_loss"), label="critic loss")
    a.set_title("Learner")
    a.legend()

    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out = folder / "progress.png"
    fig.savefig(out, dpi=110)
    print(f"saved {config.shown(out)}")


if __name__ == "__main__":
    main()
