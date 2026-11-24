"""Training curves from stage B-D logs, several runs overlaid at matching environment steps (an A/B).

    python -m f1sim.drivers.plot runs/run1 runs/run2 --out runs/compare.png
    f1sim-plot runs                                            # the newest stage's log under runs/

Each run is an output directory (its newest stage-<b|c|d>-log.jsonl) or the .jsonl itself. The chunks are binned every
`--window` environment steps: the share of learning seats alive, crashes per lap (the learners and the frozen cars),
black flags, the reward per step and Q (in the config's units: divided by the run's reward scale), entropy and
alpha; then the evals: crashes per lap and incidents per 100 km (with their 95% intervals) in the weekend race and
alone, and points per race. A log from before 2026-11-05 has no start record: its reward scale is taken as 1 and its
learning seats as `--learner-seats`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path, profile: str = "middle"):
    """(start record, chunk rows, eval rows) of one run; a log with several starts (a resumed run) gives its last one
    and every row."""
    path = Path(path)
    if path.is_dir():
        logs = sorted(path.glob("stage-*-log.jsonl"), key=lambda p: p.stat().st_mtime)
        if not logs:
            raise FileNotFoundError(f"no stage-*-log.jsonl in {path}")
        path = logs[-1]  # the newest stage's
    start, chunks, evals = {}, [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] == "start":
            start = r
        elif r["kind"] == "chunk" and profile in r["profiles"]:
            chunks.append(r)
        elif r["kind"] == "eval" and profile in r.get("profiles", {}):
            evals.append(r)
    return start, chunks, evals


def binned(chunks, profile: str, window: int, learner_seats: int, scale: float):
    """Per bin of `window` steps: (steps, series name -> values)."""
    groups: dict[int, list] = {}
    for c in chunks:
        groups.setdefault((c["env_steps"] - 1) // window, []).append(c)
    rows = []
    for b in sorted(groups):
        g = groups[b]
        m = [c["profiles"][profile] for c in g]
        f = [c["profiles"].get("frozen", {}) for c in g]
        sm = lambda rows, k: sum(x.get(k, 0.0) for x in rows)
        laps, flaps = sm(m, "laps"), sm(f, "laps")
        with_metrics = [x for x in m if "q" in x]
        mean = lambda k: float(np.mean([x[k] for x in with_metrics])) if with_metrics else np.nan
        alive = [x["alive"] for x in m if "alive" in x]
        rows.append(
            {
                "steps": g[-1]["env_steps"],
                "alive": float(np.mean(alive)) if alive else np.nan,
                "transitions": sm(m, "transitions"),
                "chunks": len(g),
                "crashes_per_lap": sm(m, "crashes") / laps if laps else np.nan,
                "frozen_crashes_per_lap": sm(f, "crashes") / flaps if flaps else np.nan,
                "hit_share": sm(m, "hit") / sm(m, "crashes") if sm(m, "crashes") and any("hit" in x for x in m) else np.nan,
                "black_flags": sm(m, "black_flags") if any("black_flags" in x for x in m) else np.nan,
                "reward": float(np.mean([x["reward_per_step"] for x in m])) / scale,
                "q": mean("q") / scale,
                "entropy": mean("entropy"),
                "alpha": mean("alpha"),
            }
        )
    # the alive share for logs without it: transitions / (steps x learning seats), steps from the chunk spacing
    for i, r in enumerate(rows):
        if np.isnan(r["alive"]):
            prev = rows[i - 1]["steps"] if i else 0
            r["alive"] = r["transitions"] / max((r["steps"] - prev) * learner_seats, 1)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage B training curves, runs overlaid.")
    ap.add_argument("runs", nargs="+", help="stage B output directories (or their stage-b-log.jsonl)")
    ap.add_argument("--profile", default="middle")
    ap.add_argument("--window", type=int, default=10_000, help="environment steps per point")
    ap.add_argument("--learner-seats", type=int, default=256, help="learning seats, for logs without a start record")
    ap.add_argument("--steps", type=int, help="only the first this many environment steps of each run")
    ap.add_argument("--out", default="runs/compare.png")
    a = ap.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("alive", "learning seats alive", False),
        ("crashes_per_lap", "crashes per lap (dashed: frozen cars)", True),
        ("black_flags", "black flags per window", False),
        ("reward", "reward per step (config units)", False),
        ("q", "Q (config units)", False),
        ("entropy", "entropy", False),
        ("alpha", "alpha", True),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    axes = axes.ravel()
    for i_run, run in enumerate(a.runs):
        start, chunks, evals = read(run, a.profile)
        if a.steps:
            chunks = [c for c in chunks if c["env_steps"] <= a.steps]
            evals = [e for e in evals if e["env_steps"] <= a.steps]
        if not chunks:
            print(f"{run}: no chunks for {a.profile}")
            continue
        scale = float(start.get("reward_scale", 1.0))
        seats = start.get("seats_per_group", [a.learner_seats])[0]
        rows = binned(chunks, a.profile, a.window, seats, scale)
        name = Path(run).name if Path(run).is_dir() else Path(run).parent.name
        colour = f"C{i_run}"
        x = np.array([r["steps"] for r in rows])
        for ax, (key, title, log) in zip(axes, panels):
            ax.plot(x, [r[key] for r in rows], color=colour, label=name)
            if key == "crashes_per_lap":
                ax.plot(x, [r["frozen_crashes_per_lap"] for r in rows], color=colour, ls="--")
            ax.set_title(title)
            if log:
                ax.set_yscale("log")
        if evals:
            ex = np.array([e["env_steps"] for e in evals])
            ev = [e["profiles"][a.profile] for e in evals]
            solo = [e.get("solo", {}).get(a.profile, {}) for e in evals]
            axes[7].plot(ex, [r["crashes_per_lap"] for r in ev], "o-", color=colour, label=f"{name} race")
            axes[7].plot(ex, [s.get("crashes_per_lap", np.nan) for s in solo], "s--", color=colour, label=f"{name} alone")
            for k, (rs, mark) in enumerate(((ev, "o-"), (solo, "s--"))):
                have = [(xx, r) for xx, r in zip(ex, rs) if "incidents_per_100km" in r]
                if have:
                    xs = np.array([h[0] for h in have])
                    ys = np.array([h[1]["incidents_per_100km"] for h in have])
                    lo = np.array([h[1]["incidents_ci"][0] for h in have])
                    hi = np.array([h[1]["incidents_ci"][1] for h in have])
                    axes[8 + k].errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=mark, color=colour, capsize=3, label=name)
            axes[10].plot(ex, [r["points_per_race"] for r in ev], "o-", color=colour, label=name)
            ref = [e["profiles"].get("reference", {}).get("points_per_race", np.nan) for e in evals]
            axes[10].plot(ex, ref, "x:", color=colour, label=f"{name} reference")
    for ax, title in zip(axes[7:11], ("eval: crashes per lap (race, alone)", "eval: incidents per 100 km, race (95%)", "eval: incidents per 100 km, alone (95%)", "eval: points per race (x: the reference)")):
        ax.set_title(title)
    axes[7].set_yscale("log")
    axes[11].axis("off")
    for ax in axes[:11]:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
