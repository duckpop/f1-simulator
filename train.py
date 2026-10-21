"""
Train a driving style with TD3: this process is the learner (GPU: networks,
optimizers, the replay buffer); rollout workers (racer/workers.py, torch-free)
each own a simulator and stream n-step transitions back. Every training car
gets its own design (random legal, or stock); worker 0 also runs the eval
rounds on the stock car that pick td3_best.pt.

    uv run --with torch --with rich train.py --profile middle --cars 1,4,8,12,16,22 --steps 200000000
    uv run --with torch --with rich train.py --profile aggressive --resume middle/best \\
        --opponents middle --opponent-speed 0.9 --lr 1e-4 --cars 4,8,12,16,22

Ctrl+C stops and saves td3_latest.pt.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import signal
import time
from collections import Counter, deque

import numpy as np

from racer import config, design
from racer.console import Reporter, dim, duration, metrics_row, num, score_text, summarize
from racer.policy import SharedPolicy
from racer.records import EVAL_FIELDS, METRICS_FIELDS, CsvLog, LapRecords, load_best, now, save_best
from racer.scoring import mine, score_round, win_rate
from racer.track import train_maps
from racer.workers import worker_main

# torch is imported inside main(): "spawn" re-imports this module in every
# worker, and the workers must stay torch-free (~500 MB RAM each).


def parse_cars(text):
    try:
        counts = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or a list like 4,8,12, got {text!r}") from None
    if not counts or min(counts) < 1:
        raise argparse.ArgumentTypeError("car counts must be >= 1")
    return sorted(set(counts))


def car_label(counts):
    """'6' or 'mix4-8-12-16-22'."""
    return str(counts[0]) if len(counts) == 1 else "mix" + "-".join(str(c) for c in counts)


def parse_args(cfg):
    p = argparse.ArgumentParser(description="Train a driving style with TD3.")
    p.add_argument("--profile", default=cfg.run.default_profile, choices=cfg.profile_names,
                   help="style to train; checkpoints in saved_models/<profile>/")
    p.add_argument("--cars", type=parse_cars, default=list(cfg.workers.default_cars),
                   help="cars per track: one count or a list (e.g. 4,8,12,16,22) every worker redraws from at "
                        "each map switch; evals use the largest")
    p.add_argument("--workers", type=int, default=cfg.worker_count, help="rollout processes")
    p.add_argument("--steps", type=int, default=cfg.run.total_steps, help="stop after this many collected env steps")
    p.add_argument("--resume", nargs="?", const=cfg.run.resume_tag, default=None, metavar="TAG",
                   help="best, latest, an update count, or PROFILE/TAG to start from another profile's driver "
                        f"(bare --resume = {cfg.run.resume_tag})")
    p.add_argument("--opponents", default="", metavar="PROFILE[:TAG],...",
                   help="frozen opponents (tag defaults to best); list a profile twice for twice the cars")
    p.add_argument("--lr", type=float, default=None, help="actor and critic learning rate (1e-4 to fine-tune)")
    p.add_argument("--best-by", choices=["auto", "distance", "points"], default="auto",
                   help="what picks td3_best.pt (auto: points with --opponents, distance without)")
    p.add_argument("--opponent-speed", type=float, default=cfg.variation.opponent_speed,
                   help="opponents' top speed in training, as a share (evals race them at full speed)")
    p.add_argument("--self-share", type=float, default=cfg.workers.self_share,
                   help="with --opponents: the trained brain's share of each grid (>= 1 car)")
    p.add_argument("--front-slots", type=int, default=cfg.workers.front_slots,
                   help="with --opponents: the trained brain's cars start in the first N grid slots, in "
                        "training and evals (0 = anywhere), e.g. 10 to practise holding places in the points")
    return p.parse_args()


def checkpoint_file(cfg, profile, tag):
    """'best' -> this profile's td3_best.pt; 'middle/best' -> another profile's."""
    tag = str(tag)
    if "/" in tag:
        other, tag = tag.split("/", 1)
        cfg.profile(other)
        return cfg.checkpoint_path(other, tag)
    return cfg.checkpoint_path(profile, tag)


def load_opponents(cfg, spec):
    """'middle,defensive:latest' -> [(name, profile, actor weights)], [name@updates]."""
    from racer.agent import read_actor_weights
    out, ids = [], []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        profile, _, tag = item.partition(":")
        tag = tag or "best"
        try:
            cfg.profile(profile)
        except ValueError as e:
            raise SystemExit(f"--opponents: {e}") from None
        path = cfg.checkpoint_path(profile, tag)
        if not path.exists():
            raise SystemExit(f"--opponents: no checkpoint at '{path}' - train '{profile}' first, or pick another tag")
        params, data = read_actor_weights(path, cfg)
        name = profile if tag == "best" else f"{profile}:{tag}"
        out.append((name, profile, params))
        ids.append(f"{name}@{data['model'].get('total_it', '?')}")
    return out, ids


def run_signature(cfg, profile, cars, best_by, opponent_ids, eval_grids):
    """Settings that make eval scores comparable; a stored best only counts
    when they match exactly."""
    sp = cfg.spawn
    sig = {"obs_dim": cfg.obs_dim, "layout": cfg.layout, "hidden": cfg.td3.hidden, "v_max": cfg.v_max,
           "lidar_range": cfg.sensors.lidar_range, "track_preview_m": list(cfg.sensors.track_preview_m),
           "design": {k: v for k, v in cfg.design.as_dict().items() if k != "designs_file"},
           "physics": {k: cfg.grip[k] for k in ("steering_scaled", "traction_circle", "combined_floor")},
           "episode_steps": cfg.episode_steps, "cars": cars,
           "profile": profile, "map_scale": cfg.sim.map_scale, "eval_maps": list(cfg.eval.maps),
           "score": "mean-distance" if best_by == "distance" else ["place-points", cfg.profile(profile).eval_bonus.as_dict()],
           "grid": [sp.grid_slot_gap, sp.grid_columns, sp.grid_column_gap, sp.grid_max_turn_deg, sp.grid_clear_ahead],
           "launch_s": sp.launch_s, "tow": cfg.tow.as_dict(), "drs": cfg.drs.as_dict(), "ers": cfg.ers.as_dict(),
           "engine": cfg.engine.as_dict()}
    if opponent_ids:
        sig.update(opponents=list(opponent_ids), self_share=cfg.workers.self_share, eval_grids=eval_grids,
                   front_slots=cfg.workers.front_slots)
    return json.loads(json.dumps(sig))


def ram_usage(workers):
    """(MB used by this process + its workers, MB the system still has available)."""
    def rss_mb(pid):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        return 0.0
    used = rss_mb(os.getpid()) + sum(rss_mb(p.pid) for p in workers if p.is_alive())
    try:
        with open("/proc/meminfo") as f:
            avail = next((int(line.split()[1]) / 1024 for line in f if line.startswith("MemAvailable:")), None)
    except OSError:
        avail = None
    return used, avail


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _interrupt)          # a kill saves td3_latest.pt like Ctrl+C
    base = config.load()
    args = parse_args(base)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not 0.0 < args.self_share <= 1.0:
        raise SystemExit("--self-share must be in (0, 1]")
    if args.front_slots < 0:
        raise SystemExit("--front-slots must be >= 0")
    overrides = {"run.total_steps": args.steps, "workers.count": args.workers,
                 "variation.opponent_speed": args.opponent_speed, "workers.self_share": args.self_share,
                 "workers.front_slots": args.front_slots}
    if args.lr is not None:
        overrides.update({"td3.lr_actor": args.lr, "td3.lr_critic": args.lr})
    cfg = base.override(overrides)
    profile = args.profile
    prof = cfg.profile(profile)
    run_dir = cfg.profile_dir(profile)
    cars = args.cars
    eval_cars = max(cars)
    label = car_label(cars)
    maps = train_maps(cfg)
    td3, wk = cfg.td3, cfg.workers

    import torch
    from racer.agent import TD3, load_checkpoint, save_checkpoint

    opponents, opponent_ids = load_opponents(cfg, args.opponents) if args.opponents else ([], [])
    best_by = args.best_by if args.best_by != "auto" else ("points" if opponents else "distance")
    if best_by == "points" and not opponents:
        raise SystemExit("--best-by points needs --opponents: copies of one driver always share the same points")
    eval_grids = max(1, cfg.eval.grids) if opponents else 1
    out = Reporter(cfg, profile, best_by, eval_cars)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if td3.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    agent = TD3(cfg, device)
    resumed = args.resume is not None
    resume_text = "fresh start"
    if resumed:
        path = checkpoint_file(cfg, profile, args.resume)
        if not path.exists():
            raise SystemExit(f"--resume: no checkpoint at '{path}'")
        data, model = load_checkpoint(path, cfg, map_location=device)
        agent.load_state_dicts(model)
        meta = data.get("meta", {})
        details = [f"{meta['cars']} car(s)" if "cars" in meta else None,
                   f"eval {score_text(meta['score'], meta.get('best_by', 'distance'))}" if "score" in meta else None,
                   f"{num(agent.total_it)} updates",
                   f"{num(meta['env_steps'])} steps" if "env_steps" in meta else None]
        resume_text = (f"resumed from {path.relative_to(cfg.save_dir)} "
                       + dim("(" + ", ".join(d for d in details if d) + ")"))
    agent.set_lr(actor=td3.lr_actor, critic=td3.lr_critic)

    ctx = mp.get_context("spawn")
    shared = SharedPolicy(cfg, ctx)
    shared.publish(agent.actor.numpy_params())
    transition_queue = ctx.Queue(maxsize=max(8, 4 * args.workers))
    stats_queue = ctx.Queue()
    stop_event = ctx.Event()
    global_steps = ctx.Value("q", 0)

    eb = prof.eval_bonus
    dz = cfg.design
    gpu = f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""
    v = cfg.variation
    out.header([
        ("Run", f"{num(args.steps)} steps · {resume_text}"),
        ("Profile", f"{profile} · {prof.description} · checkpoints in {config.shown(run_dir)}/"),
        ("Maps", (f"{len(maps)} in rotation, each worker switches map and car count every "
                  f"~{num(cfg.sim.map_switch_car_steps)} car-steps" if len(maps) > 1 else f"{maps[0]} only")
                 + f" · scale {cfg.sim.map_scale:g} · grip ±{v.grip:.0%}"
                 + (f", mass ±{v.mass:.0%}" if v.mass else "") + f", pace −0–{v.pace_spread:.0%}"),
        ("Cars", (f"{args.workers} workers × {cars[0]} car(s)" if len(cars) == 1 else
                  f"{args.workers} workers, car count redrawn from {', '.join(map(str, cars))} at every map switch")
                 + (f" · races {cfg.sim.train_race_s[0]:g}–{cfg.sim.train_race_s[1]:g} s (evals "
                    f"{cfg.episode_steps * cfg.sim.control_dt:g} s)" if cfg.train_race_steps[0] != cfg.train_race_steps[1]
                    else f" · {cfg.train_race_steps[0] * cfg.sim.control_dt:g} s races")
                 + f" · rolling starts {cfg.spawn.rolling_probability:.0%}, then grid {cfg.spawn.grid_probability:.0%} "
                   f"({cfg.spawn.grid_probability_big_field:.0%} with {cfg.spawn.big_field}+ cars) · "
                   f"{cfg.spawn.launch_s:g} s launch"),
        *([("Rivals", f"{', '.join(n for n, _, _ in opponents)} (frozen) · this brain drives "
                      f"{wk.self_share:.0%} of each grid (≥1 car)"
                      + (f", starting in the first {wk.front_slots} slots" if wk.front_slots else "")
                      + f" · opponents at {v.opponent_speed:.0%} top speed "
                      f"in training, full speed in evals")] if opponents else []),
        ("Designs", f"every training car its own: stock {dz.train_stock_share:.0%}, else uniformly random legal "
                    f"({', '.join(dz.categories)} {dz.levels[0]}–{dz.levels[1]}, ≤{dz.max_points} points, "
                    f"±{dz.step:.0%}/level, capped ±{dz.max_change:.0%}){', opponents too' if opponents else ''} · "
                    f"evals on the stock car {design.text(dz.stock)}"),
        ("Eval", f"{eval_cars} car(s) on {', '.join(cfg.eval.maps)}"
                 + (f" × {eval_grids} grids" if eval_grids > 1 else "")
                 + f" every {num(cfg.eval.every_publishes * wk.publish_every_updates)} updates · best by "
                 + ("race points" if best_by == "points" else "mean distance")
                 + f" → td3_best.pt + td3_best_{label}cars.pt"
                 + (" · per finisher: " + ", ".join(p for p in (
                     f"place points ×{eb.place_weight:g}" if eb.place_weight else "",
                     f"F1 points ×{eb.f1_weight:g}" if eb.f1_weight else "",
                     f"{eb.pass_bonus:+g}/pass on an opponent" if eb.pass_bonus else "",
                     f"{eb.crashed_pass_bonus:+g}/pass also if it crashes later" if eb.crashed_pass_bonus else "",
                     f"{-eb.passed_penalty:+g}/time passed" if eb.passed_penalty else "",
                     f"{eb.fastest_lap_bonus:+g} fastest lap" if eb.fastest_lap_bonus else "",
                     f"{eb.pace_bonus:+g} faster than the opponents" if eb.pace_bonus else "",
                     f"never below {eb.finish_floor:g}") if p)
                    if best_by == "points" else "")),
        ("Device", f"{device}{gpu}" + (" · TF32" if td3.allow_tf32 and device.type == "cuda" else "")),
        ("Car", f"{1 / cfg.sim.control_dt:.0f} Hz control (steer, throttle, brake, DRS, ERS) · stock top speed "
                f"{cfg.v_max:g} m/s ({cfg.v_max * 3.6:.0f} km/h) · lidar {cfg.sensors.lidar_beams} beams / "
                f"{cfg.sensors.lidar_range:g} m · preview to {cfg.sensors.track_preview_m[-1]:g} m · "
                + ("steering scaled to the grip limit · " if cfg.grip.steering_scaled else "")
                + (f"traction circle (≥{cfg.grip.combined_floor:.0%} lateral grip) · " if cfg.grip.traction_circle
                   else "")
                + (f"engine braking {cfg.engine.brake_share:.0%} of full braking · " if cfg.engine.brake_share else "")
                + f"DRS in zones · ERS {cfg.ers.capacity_s:g} s battery"),
        ("Network", f"obs {cfg.obs_dim} ({len(cfg.obs_beams)} beams + closing speeds, 7 state, 2 race, 3 DRS, 2 ERS, "
                    f"3 mirrors, {len(cfg.sensors.track_preview_m)} preview, 4 alongside, "
                    f"{len(cfg.design.categories)} design) · actions {cfg.action_dim} · hidden {td3.hidden} · "
                    f"{cfg.layout}"),
        ("Learning", f"lr {td3.lr_actor:g}" + (f"/{td3.lr_critic:g}" if td3.lr_critic != td3.lr_actor else "")
                     + f" · replay {num(td3.replay_size)} on {device.type}"
                     + (f" + up to {num(td3.host_replay_size)} in RAM (≥{td3.host_replay_margin_gb:g} GB left free)"
                        if agent.memory.mixed else "") + f" · batch {td3.batch} · "
                       f"{td3.updates_per_transition:g} updates/step · {td3.n_step}-step returns · γ {td3.gamma:g} "
                       f"(~{cfg.sim.control_dt / (1 - td3.gamma):.0f} s horizon)"),
        ("Noise", f"σ {td3.exploration_sigma[0]:g}–{td3.exploration_sigma[1]:g} per car per episode (log-uniform) · "
                  f"AR(1) correlation {td3.exploration_correlation:g}"),
    ])

    # One BLAS thread per worker: numpy's threads fighting over the cores made
    # workers ~500x slower. Set before spawning so the workers' numpy sees it.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    workers = []
    for wid in range(args.workers):
        p = ctx.Process(target=worker_main, daemon=True,
                        args=(wid, cfg, shared, transition_queue, stats_queue, stop_event, global_steps,
                              1000 + 7919 * wid, resumed, cars, wid == 0, profile, opponents, eval_grids))
        p.start()
        workers.append(p)
    out.start_dashboard()

    received = 0
    budget = 0.0
    learning_starts = max(td3.resume_warmup_steps if resumed else td3.warmup_steps, td3.batch)
    resume_it = agent.total_it
    critic_only_until = agent.total_it + (td3.resume_critic_only_updates if resumed else 0)

    def lr_ramp(it):
        """Share of the actor's lr in use after a resume's critic-only phase (1% steps)."""
        if not resumed or td3.actor_lr_ramp_updates <= 0:
            return 1.0
        return min(1.0, max(0.0, round((it - critic_only_until) / td3.actor_lr_ramp_updates, 2)))
    ramp_set = 1.0

    signature = run_signature(cfg, profile, cars, best_by, opponent_ids, eval_grids)
    best_meta_path = run_dir / "best_meta.json"
    best_path = cfg.checkpoint_path(profile, "best")
    best_eval, best_matched, had_meta = load_best(best_meta_path, signature)
    if had_meta and not best_matched:
        out.note("the stored best was scored under different settings - not comparable, ignored")
    elif best_matched:
        out.info(f"best eval of an earlier run: {score_text(best_eval, best_by)} - td3_best.pt is only replaced "
                 f"if this run beats it")
    best_eval_at = None
    last_eval, last_eval_maps = None, {}
    baseline_pending = resumed
    run_id = now()
    metrics_log = CsvLog(run_dir / "metrics.csv", METRICS_FIELDS, enabled=cfg.run.write_logs)
    eval_log = CsvLog(run_dir / "evals.csv", EVAL_FIELDS, enabled=cfg.run.write_logs)
    lap_records = LapRecords(run_dir / "lap_records.json", enabled=cfg.run.write_logs)
    new_records = []
    recent = deque(maxlen=200)
    recent_duels = deque(maxlen=100)
    traffic_steps = Counter()
    worker_map = {}
    episodes = 0
    workers_ready = 0
    t_start = time.time()
    t_log = t_view = t_start
    ram_warned = 0.0
    rate_samples = deque([(t_start, 0, agent.total_it)])
    save_every = cfg.run.save_every_updates
    next_save = (agent.total_it // save_every + 1) * save_every
    run_periodic = deque()

    def offer_laps(st, source):
        if st["lap_times"]:
            lap = min(st["lap_times"])
            old = lap_records.offer(st["map"], lap, source=source, cars=st["cars"], updates=agent.total_it)
            if old is not False:
                new_records.append((st["map"], lap, None if old is None else old["lap_s"]))

    def handle_eval(st):
        nonlocal baseline_pending, best_eval, best_eval_at, last_eval, last_eval_maps
        elapsed = time.time() - t_start
        rnd = score_round(st["results"], prof.eval_bonus, bool(opponents))
        score = rnd["round_points"] if best_by == "points" else rnd["round_distance"]
        per_map = rnd["points"] if best_by == "points" else rnd["distance"]
        last_eval, last_eval_maps = score, per_map
        # The baseline round (a resume's loaded driver) raises the bar; it's
        # saved only if this profile has no best yet, or it beats a stored best
        # scored under the same settings.
        baseline = baseline_pending and st["policy_version"] <= 1
        if baseline:
            baseline_pending = False
        beats = score > best_eval
        new_best = beats and (best_matched or not baseline or not best_path.exists())
        for race in rnd["races"]:
            offer_laps(race["result"], "eval")
        for m, rs in rnd["by_map"].items():
            res = [x["result"] for x in rs]
            laps = [t for r in res for t in r["lap_times"]]
            eval_log.write(dict(
                run=run_id, wall_time=now(), elapsed_s=elapsed, env_steps=received, updates=agent.total_it, map=m,
                cars=res[0]["cars"], grids=len(res), learner_cars=sum(res[0]["learner"]),
                opponents=" ".join(n for n, _, _ in opponents) or None,
                median_distance_m=float(np.median([d for r in res for d in mine(r, "progress_m")])),
                mean_distance_m=rnd["distance"][m],
                survived=float(np.mean([sum(p is not None for p in mine(r, "positions")) for r in res])),
                overtakes=float(np.mean([sum(mine(r, "overtakes")) for r in res])),
                passed=float(np.mean([sum(mine(r, "passed")) for r in res])),
                passed_by_opp=float(np.mean([x["counts"]["passed_by_opp"] for x in rs])) if rs[0]["counts"] else None,
                avg_speed_ms=float(np.mean([x for r in res for x in mine(r, "avg_speed")])),
                best_lap_s=min(laps) if laps else None, points_per_car=rnd["points"].get(m),
                round_distance_m=rnd["round_distance"], round_points=rnd["round_points"], best_by=best_by,
                new_best=int(new_best), win_rate_vs_opponents=win_rate([r["duels"] for r in res])[0]))
        saved = []
        if new_best:
            meta = {"score": score, "best_by": best_by, "per_map": per_map, "distance": rnd["round_distance"],
                    "updates": agent.total_it, "env_steps": received, "cars": label, "eval_cars": eval_cars,
                    "opponents": [n for n, _, _ in opponents]}
            saved = ["td3_best.pt", f"td3_best_{label}cars.pt"]
            save_checkpoint(agent, [best_path, cfg.checkpoint_path(profile, f"best_{label}cars")], meta,
                            actor_params=st["actor_params"])
            save_best(best_meta_path, dict(meta, signature=signature))
        if beats:                                    # a baseline that isn't saved still raises the bar
            best_eval, best_eval_at = score, elapsed
        out.eval_round(rnd, score, elapsed, new_best, saved, baseline)

    try:
        while received < args.steps:
            # 1. Train if updates are owed, otherwise pull more data.
            if received >= learning_starts and len(agent.memory) >= td3.batch and budget >= 1.0:
                it_before = agent.total_it
                ramp = lr_ramp(agent.total_it)
                if ramp != ramp_set:
                    agent.set_lr(actor=td3.lr_actor * ramp)
                    ramp_set = ramp
                while budget >= 1.0 and agent.total_it - it_before < 64:
                    done = agent.train_block(update_actor=agent.total_it >= critic_only_until)
                    if done == 0:
                        break
                    budget -= done
                if agent.total_it // wk.publish_every_updates != it_before // wk.publish_every_updates:
                    shared.publish(agent.actor.numpy_params())
            else:
                try:
                    chunk = transition_queue.get(timeout=0.5)
                except queue.Empty:
                    chunk = None
                if chunk is not None:
                    agent.memory.add_batch(*chunk)
                    received += len(chunk[0])
                    if received >= learning_starts:
                        budget += len(chunk[0]) * td3.updates_per_transition

            # 2. Episode stats.
            while True:
                try:
                    st = stats_queue.get_nowait()
                except queue.Empty:
                    break
                if "ready" in st:
                    workers_ready += 1
                    if workers_ready == args.workers:
                        out.ok(f"{workers_ready}/{args.workers} workers ready - Ctrl+C stops and saves")
                    continue
                episodes += 1
                if st["eval_round"]:
                    handle_eval(st)
                    continue
                offer_laps(st, "training")
                worker_map[st["worker"]] = st["map"]
                traffic_steps[st["cars"]] += st["car_steps"]
                if st["duels"]:
                    recent_duels.append(st["duels"])
                for i in np.flatnonzero(st["learner"]):
                    recent.append(dict(ret=st["returns"][i], progress=st["progress_m"][i], laps=st["laps"][i],
                                       reason=st["reasons"][i], overtakes=st["overtakes"][i],
                                       passed=st["passed"][i], passed_by_opp=st["passed_by_opp"][i],
                                       avg_speed=st["avg_speed"][i], max_speed=st["max_speed"][i], cars=st["cars"],
                                       tow=st["tow_frac"][i], close_calls=st["close_calls"][i],
                                       purple=st["purple"][i], drs=st["drs_frac"][i], ers=st["ers_frac"][i],
                                       battery=st["battery"][i]))

            # 3. Periodic checkpoints (this run's last keep_last kept).
            if agent.total_it >= next_save:
                path = cfg.checkpoint_path(profile, agent.total_it)
                save_checkpoint(agent, [path, cfg.checkpoint_path(profile, "latest")], {"env_steps": received})
                out.saved([path.name, "td3_latest.pt"])
                next_save += save_every
                run_periodic.append(path)
                while cfg.run.keep_last and len(run_periodic) > cfg.run.keep_last:
                    try:
                        run_periodic.popleft().unlink()
                    except OSError:
                        pass

            # 4. Status and logs.
            t = time.time()
            write_due = t - t_log >= cfg.run.log_every_s
            if write_due or (out.live and t - t_view >= cfg.run.dashboard_refresh_s):
                t_view = t
                rate_samples.append((t, received, agent.total_it))
                while len(rate_samples) > 2 and t - rate_samples[0][0] > 30:
                    rate_samples.popleft()
                t0, steps0, it0 = rate_samples[0]
                sps = (received - steps0) / max(t - t0, 1e-6)
                ups = (agent.total_it - it0) / max(t - t0, 1e-6)
                ram = ram_usage(workers)
                if ram[1] is not None and ram[1] < 1536 and t - ram_warned > 600:
                    ram_warned = t
                    out.note(f"only {ram[1] / 1024:.1f} GB RAM free - the system may start killing processes")
                if new_records:
                    out.lap_records(new_records)
                    new_records.clear()
                sm = summarize(recent) if recent else None
                losses = agent.diagnostics()
                out.status(dict(
                    elapsed=t - t_start, received=received, total_steps=args.steps, steps_per_s=sps,
                    updates=agent.total_it, updates_per_s=ups, episodes=episodes, buffer_len=len(agent.memory),
                    buffer_cap=agent.memory.capacity, buffer_ram=agent.memory.ram_rows, warmup_left=max(0, learning_starts - received),
                    warmup_total=learning_starts, warmup_policy=resumed,
                    critic_only_left=max(0, critic_only_until - agent.total_it),
                    critic_only_total=critic_only_until - resume_it, lr_ramp=lr_ramp(agent.total_it),
                    losses=losses, summary=sm, duels=recent_duels, traffic_steps=dict(traffic_steps),
                    maps_now=[worker_map[w] for w in sorted(worker_map)], maps=maps,
                    lap_records=lap_records.records, ram=ram, best_eval=best_eval, best_eval_at=best_eval_at,
                    last_eval=last_eval, last_eval_maps=last_eval_maps))
                if write_due:
                    metrics_log.write(metrics_row(
                        run=run_id, elapsed=t - t_start, received=received, updates=agent.total_it, steps_per_s=sps,
                        updates_per_s=ups, summary=sm, best_eval=best_eval, best_by=best_by, losses=losses,
                        ram=ram, duels=recent_duels))
                    t_log = t
    except KeyboardInterrupt:
        if not out.live:
            print()
        out.note("interrupted - saving td3_latest.pt ...")
    finally:
        stop_event.set()
        try:
            save_checkpoint(agent, [cfg.checkpoint_path(profile, "latest")], {"env_steps": received})
            deadline = time.time() + 5            # unblock workers stuck on a full queue
            while time.time() < deadline and any(p.is_alive() for p in workers):
                try:
                    transition_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            for p in workers:
                if p.is_alive():
                    p.terminate()
        best_text = f" · best eval {score_text(best_eval, best_by)}" if best_eval > float("-inf") else ""
        out.finished(f"Training finished after {duration(time.time() - t_start)} · {num(received)} steps · "
                     f"{num(agent.total_it)} updates{best_text}", f"checkpoints in {config.shown(run_dir)}/")


if __name__ == "__main__":
    main()
