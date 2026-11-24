"""GPU benchmark for planning runs on another machine (e.g. a cluster A100), ~15 minutes. Nothing trains for real.

    python scripts/benchmark.py                      # generated tracks; results in runs/benchmark-<host>-<time>.{json,txt}
    python scripts/benchmark.py --tracks tracks      # the track pool's eval tracks instead
    python scripts/benchmark.py --quick              # a ~3-minute check that it runs

It measures, with the config's settings (shared critic encoder, bf16, lidar steps):
1. the gradient step at batch 512-4096 (and at 512 without the speed options);
2. stage A chunks (one car per race) at 64-512 races, learning one gradient step per environment step;
3. stage B chunks (22 cars per race, frozen snapshot pool) at 16-128 races, without learning and with learning
   at the config's reuse;
4. the eval weekend's speed (one-lap qualifying) at 5 x 2 and 10 x 2 races;
and the GPU, versions and peak memory. Results are written after every measurement, so a cut-short run keeps what
it measured. Send both files back to compare with the laptop.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1env import generatemap  # noqa: E402
from f1env.sensors import make_bank  # noqa: E402

from f1sim import loop, tqc  # noqa: E402
from f1sim import replay as rp  # noqa: E402
from f1sim.drivers import env as de  # noqa: E402
from f1sim.drivers import season, stage_b  # noqa: E402
from f1sim.drivers.config import load  # noqa: E402
from f1sim.drivers.tracks import eval_tracks  # noqa: E402


def git(path, *args):
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_info():
    try:
        q = "name,memory.total,driver_version,power.limit,clocks.max.sm"
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi not found"


def peak_mb():
    try:
        stats = jax.devices()[0].memory_stats() or {}
        return round(stats.get("peak_bytes_in_use", 0) / 2**20)
    except Exception:  # noqa: BLE001 - not every backend reports it
        return None


def timed(fn, reps):
    """(first call incl. compile, best of `reps` later calls) in s."""
    t = time.perf_counter()
    jax.block_until_ready(fn())
    first = time.perf_counter() - t
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t)
    return first, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tracks", help="track pool directory (default: generated tracks)")
    ap.add_argument("--quick", action="store_true", help="a few small cases only (~3 min): checks it runs")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--budget-min", type=float, default=16.0, help="stop starting new measurements after this long")
    a = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    host = socket.gethostname().split(".")[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath, tpath = out / f"benchmark-{host}-{stamp}.json", out / f"benchmark-{host}-{stamp}.txt"
    t_start = time.time()
    s = load("middle", "laptop")
    sb = dict(s.stage_b)
    res = {
        "host": host, "when": stamp, "gpu": gpu_info(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
        "python": platform.python_version(), "f1sim_git": git(ROOT, "rev-parse", "HEAD"), "f1sim_dirty": bool(git(ROOT, "status", "--porcelain", "--", "src", "configs")),
        "f1env": str(Path(sys.modules["f1env"].__file__).parent), "f1env_git": git(Path(sys.modules["f1env"].__file__).parents[2], "rev-parse", "HEAD"),
        "settings": {"shared_encoder": s.tqc.shared_encoder, "bf16": s.tqc.encoder.bf16, "march_steps": s.env.march_steps, "critics": s.tqc.critics, "critic_hidden": s.tqc.critic_hidden, "batch": s.tqc.batch, "reuse_b": sb.get("reuse")},
        "learning_ms": {}, "stage_a": {}, "stage_b": {}, "eval": {}, "notes": [],
    }
    lines = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    def save():
        res["minutes"] = round((time.time() - t_start) / 60, 1)
        res["peak_gpu_mb"] = peak_mb()
        jpath.write_text(json.dumps(res, indent=1, default=str))
        tpath.write_text("\n".join(lines) + "\n")

    def over():
        if (time.time() - t_start) / 60 > a.budget_min:
            res["notes"].append("time budget reached: later measurements skipped")
            return True
        return False

    say(f"benchmark on {host}: {res['gpu']} | jax {res['jax']} | f1sim {res['f1sim_git'][:10]} f1env {res['f1env_git'][:10]}")
    say(f"settings: {res['settings']}")
    save()
    reps = 1 if a.quick else 3

    # 1. the gradient step
    say("\n1. gradient step (critics, actor, temperature), ms per step")
    slots = 96
    buf = rp.new(8000, slots, de.OBS, de.ACTIONS, de.storage_codec())
    add = jax.jit(lambda b, o: rp.add(b, o, jnp.zeros((slots, de.ACTIONS)), jnp.zeros(slots), jnp.ones(slots, bool), jnp.zeros(slots, jnp.int8)), donate_argnums=0)
    for i in range(300):
        buf = add(buf, jax.random.uniform(jax.random.PRNGKey(i), (slots, de.OBS)))
    cases = [(512, {}), (1024, {}), (2048, {}), (4096, {}), (512, {"shared_encoder": False, "bf16": False})]
    for batch, extra in cases[:2] if a.quick else cases:
        if over():
            break
        cfg = s.tqc._replace(batch=batch, shared_encoder=extra.get("shared_encoder", s.tqc.shared_encoder))
        cfg = cfg._replace(encoder=cfg.encoder._replace(bf16=extra.get("bf16", cfg.encoder.bf16)))
        agent = tqc.init(jax.random.PRNGKey(1), cfg)
        n = 50
        f = jax.jit(lambda ag, b, k, cfg=cfg: loop.learn(ag, cfg, b, k, n, jnp.float32(1.0)))
        _, best = timed(lambda: f(agent, buf, jax.random.PRNGKey(2)), reps)
        label = f"batch {batch}" + (" without the speed options" if extra else "")
        res["learning_ms"][label] = round(1000 * best / n, 2)
        say(f"  {label:<34} {1000 * best / n:7.2f} ms  ({batch * n / best:,.0f} samples/s)")
        save()

    # tracks
    tracks = eval_tracks(a.tracks, "normal", 10) if a.tracks else [generatemap(seed=1000 + k) for k in range(10)]
    bank = make_bank(tracks)
    steps = 200 if a.quick else 400

    # 2. stage A: one car per race, one gradient step per environment step
    say("\n2. stage A chunk (one car per race, learning), environment steps/s and car-steps/s")
    for races in ([64, 256] if a.quick else [64, 128, 256, 512]):
        if over():
            break
        ecfg = s.env._replace(races=races, cars=1)
        w = jnp.asarray(s.weights)
        st, obs = de.compiled("reset", ecfg)(bank, jax.random.PRNGKey(3))
        rows = rp.rows_for(4_000_000, races)
        carry = loop.Carry(st, obs.reshape(races, de.OBS), tqc.init(jax.random.PRNGKey(4), s.tqc), rp.new(rows, races, de.OBS, de.ACTIONS, de.storage_codec()), jax.random.PRNGKey(5), jnp.zeros((), jnp.int32))
        from f1sim.drivers.train import stage_a_step
        from functools import partial

        chunk = loop.make_chunk(partial(stage_a_step, cfg=ecfg, w=w), s.tqc, steps, 1, 0, 0, 0, 0)
        state = {"c": carry}

        def run():
            state["c"], out_ = chunk(state["c"], bank)
            return state["c"].steps

        first, best = timed(run, reps)
        res["stage_a"][f"{races} races"] = {"steps_per_s": round(steps / best, 1), "car_steps_per_s": round(races * steps / best), "compile_s": round(first - best, 1)}
        say(f"  {races:4} races: {steps / best:7.1f} steps/s, {races * steps / best:9,.0f} car-steps/s  (compile {first - best:.0f} s)")
        del state, carry, chunk
        save()

    # 3. stage B: 22 cars per race, Middle learning in 16 seats, 6 frozen
    say("\n3. stage B chunk (22 cars per race, 16 learning, 8 snapshots), environment steps/s and learner car-steps/s")
    seats = stage_b.seat_layout(22, 16, 1)
    stored = [np.flatnonzero(seats == 0)]
    frozen = int((seats < 0).sum())
    cfg_b = s.tqc._replace(lr=float(sb["lr"]), actor_lr=float(sb["actor_lr"]), kappa=float(sb["kappa"]))
    for races in ([16, 64] if a.quick else [16, 32, 64, 128]):
        for learning in (False, True):
            if over():
                break
            env_b = s.env._replace(races=races, cars=22, quali_share=sb["quali_share"], learners=tuple(bool(x) for x in seats >= 0), keep=tuple(bool(x) for x in seats >= 0),
                                   field_sizes=tuple(sb["field_sizes"]), spread_share=float(sb["spread_share"]), learners_first=True, race=de.stage_race(s.env.race, sb))
            slots_b = races * len(stored[0])
            updates = stage_b.updates_for(sb, 1, slots_b, cfg_b.batch)
            buf_b = rp.new(rp.rows_for(4_000_000, slots_b), slots_b, de.OBS, de.ACTIONS, de.storage_codec())
            if learning:  # rows to sample from
                add_b = jax.jit(lambda b, o: rp.add(b, o, jnp.zeros((slots_b, de.ACTIONS)), jnp.zeros(slots_b), jnp.ones(slots_b, bool), jnp.zeros(slots_b, jnp.int8)), donate_argnums=0)
                for i in range(100):
                    buf_b = add_b(buf_b, jax.random.uniform(jax.random.PRNGKey(i), (slots_b, de.OBS)))
            agent = tqc.init(jax.random.PRNGKey(6), cfg_b)
            st, obs = de.compiled("reset", env_b)(bank, jax.random.PRNGKey(7))
            league = stage_b.League(st, obs, (agent,), (buf_b,), stage_b.stack([agent.actor] * 8), stage_b.assign_pool(jax.random.PRNGKey(8), races, frozen, 8), jax.random.PRNGKey(9), jnp.zeros((), jnp.int32))
            plan = loop.Schedule(0, 0, 0) if learning else loop.Schedule(10**9, 10**9, 10**9)
            chunk = stage_b.make_chunk([cfg_b], env_b, seats, s.weights[None], steps, updates, plan, stored)
            state = {"l": league}

            def run():
                state["l"], out_ = chunk(state["l"], bank)
                return state["l"].steps

            first, best = timed(run, reps)
            key = f"{races} races, " + (f"learning ({updates} gradient steps/step)" if learning else "no learning")
            res["stage_b"][key] = {"steps_per_s": round(steps / best, 1), "learner_car_steps_per_s": round(slots_b * steps / best), "compile_s": round(first - best, 1)}
            say(f"  {key:<40} {steps / best:7.1f} steps/s, {slots_b * steps / best:9,.0f} learner car-steps/s  (compile {first - best:.0f} s)")
            del state, league, chunk, buf_b
            save()

    # 4. the eval weekend (one-lap qualifying: it runs to its set end whatever the cars do)
    say("\n4. eval weekend, one-lap qualifying session, environment steps/s")
    actors = [tqc.init(jax.random.PRNGKey(k), s.tqc).actor for k in range(2)]
    for n_tracks, seeds in ([(5, 2)] if a.quick else [(5, 2), (10, 2)]):
        if over():
            break
        env_e = season.weekend_env(tracks[:n_tracks], s.preset["eval_laps"], sb)
        times = {}
        orig = season.run_session

        def timed_session(env, session, *args, **kw):
            t = time.time()
            r = orig(env, session, *args, **kw)
            jax.block_until_ready(r[1].laps)
            times.setdefault(session.kind, []).append((r[2], time.time() - t))
            return r

        season.run_session = timed_session
        try:
            for _ in range(2):  # the first compiles
                season.eval_weekend(env_e, actors[:1], s.tqc, seeds, names=("middle",), reference=actors[1], march_steps=s.env.march_steps)
        finally:
            season.run_session = orig
        n, t = times["one-lap"][-1]
        res["eval"][f"{n_tracks} tracks x {seeds}"] = {"races": n_tracks * seeds, "steps_per_s": round(n / t, 1), "qualifying_steps": n}
        say(f"  {n_tracks} tracks x {seeds} ({n_tracks * seeds} races): {n / t:7.1f} steps/s")
        save()

    say(f"\ndone in {(time.time() - t_start) / 60:.1f} min; peak GPU memory {peak_mb()} MB")
    save()
    print(f"results: {jpath}\n         {tpath}")


if __name__ == "__main__":
    main()
