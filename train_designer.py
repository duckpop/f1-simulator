"""
Train the designer bot: one general double-DQN designer that fits every
team's car to its two drivers over a season. This process is the learner (a
small network on the CPU); season workers (racer/season.py, torch-free) race
seasons with the shared designer weights, epsilon-greedy, and send every
team's decisions back. Test workers race the fixed test seasons with the
designer on half the field - against teams keeping their random starting car
and against stock cars - and the best score is saved as designer_best.pt.

    uv run --with torch train_designer.py
    uv run --with torch train_designer.py --resume --workers 10

The drivers are the [season] profiles' best car-aware checkpoints (--drivers);
train them first (project_v2.md §8). Ctrl+C stops and saves designer_latest.pt.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import time
from collections import deque

import numpy as np

from racer import config, design
from racer.console import bold, cyan, dim, duration, green, num, yellow
from racer.policy import SharedPolicy, designer_shapes
from racer.records import CsvLog, load_best, now, save_best
from racer.season import TEST_RUNS, test_worker_main, train_worker_main
from racer.track import available_maps

# torch is imported inside main(): "spawn" re-imports this module in every
# worker, and the workers must stay torch-free.

METRICS_FIELDS = [
    "run", "wall_time", "elapsed_s", "decisions", "races", "updates", "decisions_per_s", "updates_per_s",
    "epsilon", "loss", "avg_q", "keep_share", "add_share", "move_share", "points_used", "best_score", "last_score",
]
TEST_FIELDS = [
    "run", "wall_time", "elapsed_s", "decisions", "updates", "round", "score", "designer_vs_keep", "rivals_keep",
    "designer_vs_stock", "rivals_stock", "points_used", "new_best",
]


def parse_args(cfg):
    d = cfg.designer
    p = argparse.ArgumentParser(description="Train the designer bot (double DQN over seasons).")
    p.add_argument("--workers", type=int, default=d.workers, help="season workers producing training data")
    p.add_argument("--test-workers", type=int, default=d.test_seasons,
                   help=f"processes racing the test seasons ({len(TEST_RUNS)} seasons per test season per check)")
    p.add_argument("--decisions", type=int, default=d.total_decisions, help="stop after this many decisions")
    p.add_argument("--resume", nargs="?", const="best", default=None, metavar="TAG",
                   help="best, latest or an update count (bare --resume = best)")
    p.add_argument("--drivers", default="", metavar="PROFILE[:TAG],...",
                   help=f"the drivers teams draw from (default: {', '.join(cfg.season.profiles)} at best)")
    p.add_argument("--lr", type=float, default=None, help="learning rate")
    return p.parse_args()


def run_signature(cfg, driver_ids):
    """Settings that make test scores comparable; a stored best only counts
    when they match exactly."""
    sig = {"layout": cfg.designer_layout, "hidden": cfg.designer.hidden, "season": cfg.season.as_dict(),
           "test_seed": cfg.designer.test_seed, "test_seasons": cfg.designer.test_seasons,
           "test_runs": [list(r) for r in TEST_RUNS], "drivers": list(driver_ids),
           "design": {k: v for k, v in cfg.design.as_dict().items() if k != "designs_file"},
           "physics": {k: cfg.grip[k] for k in ("steering_scaled", "traction_circle", "combined_floor")},
           "engine": cfg.engine.as_dict(),
           "episode_steps": cfg.episode_steps, "map_scale": cfg.sim.map_scale, "maps": sorted(available_maps(cfg))}
    return json.loads(json.dumps(sig))


def score_round(results):
    """A test round: the designer's teams' F1 points per race against rivals
    keeping their car / on stock cars, and overall (the score)."""
    def mean(key, rivals=None):
        vals = [r["season"][key] for r in results if rivals in (None, r["rivals"])]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    final = [np.asarray(r["season"]["designs"][-1])[np.asarray(r["season"]["designed"])] for r in results]
    return {"score": mean("designer_ppr"), "designer_vs_keep": mean("designer_ppr", "keep"),
            "rivals_keep": mean("rival_ppr", "keep"), "designer_vs_stock": mean("designer_ppr", "stock"),
            "rivals_stock": mean("rival_ppr", "stock"),
            "points_used": float(np.mean([d.sum(axis=1).mean() for d in final if len(d)]))}


def season_report(results, cfg, rnd, round_id):
    """The best round as text: per test season and run, every team's car at
    each race and its points there."""
    rivals_text = {"keep": "rivals keep their random starting car", "stock": "rivals race the stock car"}
    lines = [f"Designer test round {round_id} · score {rnd['score']:.2f} F1 points per team per race "
             f"(vs keep {rnd['designer_vs_keep']:.2f} : {rnd['rivals_keep']:.2f}, "
             f"vs stock {rnd['designer_vs_stock']:.2f} : {rnd['rivals_stock']:.2f})",
             f"Cars as {', '.join(cfg.design.categories)} levels; each race: car · points scored there.", ""]
    for r in sorted(results, key=lambda r: (r["index"], r["rivals"], r["half"])):
        s = r["season"]
        lines.append(f"Test season {r['index'] + 1}, {rivals_text[r['rivals']]}, designer on the "
                     f"{'odd' if r['half'] else 'even'}-numbered teams · designer {s['designer_ppr']:.2f}, "
                     f"rivals {s['rival_ppr']:.2f} pts/race")
        lines.append("  calendar: " + ", ".join(s["calendar"]))
        for t, drivers in enumerate(s["drivers"]):
            pts = [sum(race[t]) for race in s["points"]]
            who = "designer" if s["designed"][t] else r["rivals"]
            lines.append(f"  Team {t + 1:<2} ({who}) {' + '.join(drivers)} · {np.mean(pts):.1f} pts/race")
            cells = [f"{design.text(race[t]).replace('-', '')}·{p:g}" for race, p in zip(s["designs"], pts)]
            for k in range(0, len(cells), 10):
                lines.append("      " + "  ".join(cells[k:k + 10]))
        lines.append("")
    return "\n".join(lines)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _interrupt)          # a kill saves designer_latest.pt like Ctrl+C
    base = config.load()
    args = parse_args(base)
    if args.workers < 1 or args.test_workers < 1:
        raise SystemExit("--workers and --test-workers must be >= 1")
    cfg = base.override({"designer.lr": args.lr}) if args.lr is not None else base
    dq = cfg.designer
    run_dir = cfg.designer_dir

    import torch
    from racer.agent import DesignerDQN, load_designer, read_season_drivers, save_designer
    torch.set_num_threads(2)

    drivers, driver_ids = read_season_drivers(cfg, args.drivers)
    dqn = DesignerDQN(cfg)
    decisions = 0
    resume_text = "fresh start"
    if args.resume is not None:
        path = cfg.designer_path(args.resume)
        if not path.exists():
            raise SystemExit(f"--resume: no checkpoint at '{config.shown(path)}'")
        data, model = load_designer(path, cfg)
        dqn.load_state_dicts(model)
        decisions = int(data.get("meta", {}).get("decisions", 0))      # continues the epsilon schedule
        resume_text = f"resumed from {path.name} ({num(dqn.total_it)} updates, {num(decisions)} decisions)"

    ctx = mp.get_context("spawn")
    shared = SharedPolicy(cfg, ctx, designer_shapes(cfg))
    shared.publish(dqn.net.numpy_params())
    eps0, eps1 = dq.epsilon

    def epsilon_at(d):
        return eps1 + (eps0 - eps1) * max(0.0, 1.0 - d / max(1, dq.epsilon_decay_decisions))
    epsilon = ctx.Value("d", epsilon_at(decisions))
    decision_queue = ctx.Queue(maxsize=max(8, 4 * args.workers))
    task_queue = ctx.Queue()
    stats_queue = ctx.Queue()
    stop_event = ctx.Event()

    s = cfg.season
    rows = [("Run", f"{num(args.decisions)} decisions · {resume_text}"),
            ("Season", f"{s.teams} teams × 2 cars · {s.races} races (every track once, random order) · "
                       f"≤{s.max_moves} point operations per team before each race · F1 points"),
            ("Drivers", ", ".join(driver_ids) + " · each team draws 2 (repeats allowed)"),
            ("Cars", f"start uniformly random legal · {', '.join(cfg.design.categories)} levels "
                     f"{cfg.design.levels[0]}–{cfg.design.levels[1]}, ≤{cfg.design.max_points} points"),
            ("Learning", f"double DQN {cfg.designer_obs_dim} → {dq.hidden} → {dq.hidden} → {cfg.designer_ops} on the "
                         f"CPU · lr {dq.lr:g} · γ {dq.gamma:g} per race · replay {num(dq.replay_size)} · batch "
                         f"{dq.batch} · {dq.updates_per_decision:g} updates/decision · target every "
                         f"{num(dq.target_every_updates)} updates"),
            ("Explore", f"ε {eps0:g} → {eps1:g} over {num(dq.epsilon_decay_decisions)} decisions · "
                        f"{num(dq.warmup_decisions)} random first"),
            ("Workers", f"{args.workers} season workers + {args.test_workers} test workers"),
            ("Tests", f"{dq.test_seasons} fixed test seasons × {len(TEST_RUNS)} runs (designer on half the field, "
                      f"halves swapped; rivals keep their car / race stock) → designer_best.pt"),
            ("Folder", config.shown(run_dir) + "/")]
    print("\n" + cyan("━━ F1 racers · designer training " + "━" * 45))
    for label, text in rows:
        print(f"  {bold(f'{label:<9}')} {text}")
    print(cyan("━" * 78), flush=True)

    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    procs = []
    for wid in range(args.workers):
        procs.append(ctx.Process(target=train_worker_main, daemon=True,
                                 args=(wid, cfg, shared, epsilon, decision_queue, stats_queue, stop_event,
                                       1000 + 7919 * wid + decisions, drivers)))
    for k in range(args.test_workers):
        procs.append(ctx.Process(target=test_worker_main, daemon=True,
                                 args=(args.workers + k, cfg, task_queue, stats_queue, stop_event, drivers)))
    for p in procs:
        p.start()

    signature = run_signature(cfg, driver_ids)
    best_meta_path = run_dir / "best_meta.json"
    best_score, best_matched, had_meta = load_best(best_meta_path, signature)
    if had_meta and not best_matched:
        print(f"  {yellow('!')} the stored best was scored under different settings - not comparable, ignored")
    elif best_matched:
        print(f"  {dim('·')} {dim(f'best test score of an earlier run: {best_score:.2f} pts/race')}")
    run_id = now()
    metrics_log = CsvLog(run_dir / "metrics.csv", METRICS_FIELDS, enabled=cfg.run.write_logs)
    test_log = CsvLog(run_dir / "tests.csv", TEST_FIELDS, enabled=cfg.run.write_logs)
    recent = deque(maxlen=500)                # training races: (ops counts, mean points used)
    races = 0
    budget = 0.0
    ready = 0
    rounds = {}                               # round -> (params, results)
    round_id, round_open = 0, False
    last_score = None
    t_start = time.time()
    t_log = t_start
    rate = deque([(t_start, decisions, dqn.total_it)])
    save_every = dq.save_every_updates
    next_save = (dqn.total_it // save_every + 1) * save_every
    periodic = deque()
    expected = dq.test_seasons * len(TEST_RUNS)

    def finish_round(rid):
        nonlocal best_score, last_score
        params, results = rounds.pop(rid)
        rnd = score_round(results)
        last_score = rnd["score"]
        new_best = rnd["score"] > best_score
        if new_best:
            best_score = rnd["score"]
            meta = dict(rnd, round=rid, updates=dqn.total_it, decisions=decisions)
            save_designer(dqn, [cfg.designer_path("best")], meta, params=params)
            save_best(best_meta_path, dict(meta, signature=signature))
            if cfg.run.write_logs:
                (run_dir / "season_report.txt").write_text(season_report(results, cfg, rnd, rid))
        test_log.write(dict(run=run_id, wall_time=now(), elapsed_s=time.time() - t_start, decisions=decisions,
                            updates=dqn.total_it, round=rid, new_best=int(new_best), **rnd))
        star = f"  {green('★ new best → designer_best.pt, season_report.txt')}" if new_best else ""
        when, score = dim(f"[{duration(time.time() - t_start):>8}]"), bold(f"{rnd['score']:.2f}")
        print(f"  {when} test round {rid}: designer {score} pts/race · vs keep {rnd['designer_vs_keep']:.2f} : "
              f"{rnd['rivals_keep']:.2f} · vs stock {rnd['designer_vs_stock']:.2f} : {rnd['rivals_stock']:.2f} · "
              f"its cars use {rnd['points_used']:.1f} points{star}", flush=True)

    try:
        while decisions < args.decisions:
            # 1. Learn if updates are owed, otherwise pull more decisions.
            if decisions >= dq.warmup_decisions and budget >= 1.0 and len(dqn.memory) >= dq.batch:
                it_before = dqn.total_it
                while budget >= 1.0 and dqn.total_it - it_before < 64 and dqn.update():
                    budget -= 1.0
                if dqn.total_it // dq.publish_every_updates != it_before // dq.publish_every_updates:
                    shared.publish(dqn.net.numpy_params())
            else:
                try:
                    batch = decision_queue.get(timeout=0.5)
                except queue.Empty:
                    batch = None
                if batch is not None:
                    dqn.memory.add_batch(*batch)
                    decisions += len(batch[0])
                    if decisions >= dq.warmup_decisions:
                        budget += len(batch[0]) * dq.updates_per_decision
                    epsilon.value = epsilon_at(decisions)

            # 2. Race stats and test results.
            while True:
                try:
                    st = stats_queue.get_nowait()
                except queue.Empty:
                    break
                if "ready" in st:
                    ready += 1
                    if ready == len(procs):
                        print(f"  {green('✓')} {ready}/{len(procs)} workers ready - Ctrl+C stops and saves", flush=True)
                elif st.get("test"):
                    if st["round"] in rounds:
                        rounds[st["round"]][1].append(st)
                        if len(rounds[st["round"]][1]) == expected:
                            finish_round(st["round"])
                            round_open = False
                else:
                    races += 1
                    recent.append((np.asarray(st["ops"]), float(np.sum(st["designs"], axis=1).mean())))

            # 3. A test round when none is running: the current weights on every
            # test season and run (any free test worker takes the next one).
            if not round_open and decisions >= dq.warmup_decisions:
                round_id += 1
                params = [p.copy() for p in dqn.net.numpy_params()]
                rounds[round_id] = (params, [])
                for index in range(dq.test_seasons):
                    for rivals, half in TEST_RUNS:
                        task_queue.put((round_id, index, rivals, half, params))
                round_open = True

            # 4. Periodic checkpoints (this run's last keep_last kept).
            if dqn.total_it >= next_save:
                path = cfg.designer_path(dqn.total_it)
                save_designer(dqn, [path, cfg.designer_path("latest")], {"decisions": decisions})
                next_save += save_every
                periodic.append(path)
                while dq.keep_last and len(periodic) > dq.keep_last:
                    try:
                        periodic.popleft().unlink()
                    except OSError:
                        pass

            # 5. Status and logs.
            t = time.time()
            if t - t_log >= cfg.run.log_every_s:
                t_log = t
                rate.append((t, decisions, dqn.total_it))
                while len(rate) > 2 and t - rate[0][0] > 120:
                    rate.popleft()
                t0, d0, it0 = rate[0]
                dps = (decisions - d0) / max(t - t0, 1e-6)
                ups = (dqn.total_it - it0) / max(t - t0, 1e-6)
                ops = np.sum([o for o, _ in recent], axis=0) if recent else np.zeros(design.NUM_OPS)
                total = max(ops.sum(), 1)
                keep, add = ops[0] / total, ops[design.OP_FROM < 0][1:].sum() / total
                used = float(np.mean([u for _, u in recent])) if recent else None
                loss, q = dqn.last
                metrics_log.write(dict(run=run_id, wall_time=now(), elapsed_s=t - t_start, decisions=decisions,
                                       races=races, updates=dqn.total_it, decisions_per_s=dps, updates_per_s=ups,
                                       epsilon=epsilon.value, loss=loss, avg_q=q, keep_share=keep, add_share=add,
                                       move_share=1 - keep - add if recent else None, points_used=used,
                                       best_score=best_score if best_score > float("-inf") else None,
                                       last_score=last_score))
                best = f"{best_score:.2f}" if best_score > float("-inf") else "-"
                print(f"  {dim(f'[{duration(t - t_start):>8}]')} {num(decisions)} decisions ({dps:.0f}/s) · "
                      f"{num(races)} races · {num(dqn.total_it)} updates ({ups:.0f}/s) · ε {epsilon.value:.2f} · "
                      + (f"loss {loss:.3g} · Q {q:.1f} · " if loss is not None else "warm-up · ")
                      + (f"ops keep {keep:.0%} add {add:.0%} move {1 - keep - add:.0%} · cars {used:.1f} pts · "
                         if recent else "")
                      + f"test best {best}" + (f", last {last_score:.2f}" if last_score is not None else ""),
                      flush=True)
    except KeyboardInterrupt:
        print(f"\n  {yellow('!')} interrupted - saving designer_latest.pt ...", flush=True)
    finally:
        stop_event.set()
        try:
            save_designer(dqn, [cfg.designer_path("latest")], {"decisions": decisions})
            deadline = time.time() + 5            # unblock workers stuck on a full queue
            while time.time() < deadline and any(p.is_alive() for p in procs):
                try:
                    decision_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
        best = f" · best test score {best_score:.2f} pts/race" if best_score > float("-inf") else ""
        print(f"\n  {green('■')} Designer training finished after {duration(time.time() - t_start)} · "
              f"{num(decisions)} decisions · {num(dqn.total_it)} updates{best}\n"
              f"    {dim(f'checkpoints in {config.shown(run_dir)}/')}\n", flush=True)


if __name__ == "__main__":
    main()
