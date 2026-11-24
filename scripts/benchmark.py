"""The cluster checker (2026-11-24): realistic training speeds for stages A, B, C and D and the evals on this GPU, and
from them how long each stage takes and how many 24-hour jobs it needs. About 15 minutes; nothing is kept.

    python scripts/benchmark.py                 # results in runs/benchmark-<host>-<time>.json and .txt
    python scripts/benchmark.py --quick         # shorter runs (~6 min), rougher numbers

How: it runs the real trainers for a few thousand steps each, with every stage's own settings (stage A: 64 races of
one car; B: Middle in 16 seats; C: B plus the team orders; D: Smooth, Aggressive and Defensive together), on
generated tracks and from a fresh driver, into a scratch folder, and reads their steps per second once learning is
under way (the warm start shortened to get there). The evals' speed comes from their fixed-length qualifying session,
scaled by how much slower trained drivers make each step (measured on the laptop). Results are written after every
part, so a cut-short job keeps what it measured. Send both files back.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent
TRAINED_FACTOR = 0.61  # a trained field's eval step costs 1/0.61 of an untrained one's (laptop, 2026-11-24: 350 vs 575 steps/s)
EVAL_STEPS_B = 4_400 + 150_000  # stages B-D: one-lap qualifying + the race on the longest of the 5 eval tracks (eval-08, ~136k) with safety-car slack
EVAL_STEPS_A = 160_000  # stage A: 50 laps alone on the 10 eval tracks, the longest (eval-09) ~146k, plus slack
JOB_H = 24.0


def gpu_info():
    try:
        q = "name,memory.total,driver_version,clocks.max.sm"
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi not found"


def smi(fields):
    """nvidia-smi's values for these query fields, {field: value} (per GPU: the first one visible)."""
    out = {}
    for f in fields:  # one at a time: a field this driver doesn't know shouldn't lose the others
        try:
            v = subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            out[f] = v[0].strip() if v else ""
        except (OSError, subprocess.SubprocessError):
            out[f] = "nvidia-smi not found"
    return out


def gpu_specs():
    """The GPU and what this job may use of it: the card (nvidia-smi), the job's share (SLURM, cgroup limits, MIG),
    JAX's view (devices, the memory it may allocate), and measured speeds (matrix products, memory bandwidth)."""
    import jax
    import jax.numpy as jnp

    card = smi(["name", "memory.total", "memory.used", "memory.free", "compute_cap", "clocks.max.sm", "clocks.max.memory", "clocks.sm",
                "power.limit", "power.max_limit", "pcie.link.gen.max", "pcie.link.width.max", "driver_version", "mig.mode.current",
                "persistence_mode", "ecc.mode.current", "temperature.gpu", "count"])
    try:
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout)
        card["cuda_version"] = cuda.group(1) if cuda else ""
        card["mig_instances"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    job = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE",
                                          "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_JOB_NODELIST") if os.environ.get(k)}
    try:
        job["cpus_usable"] = len(os.sched_getaffinity(0))
    except AttributeError:
        job["cpus_usable"] = os.cpu_count()
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # the job's RAM limit
        try:
            v = Path(path).read_text().strip()
            job["ram_limit_gb"] = "none" if v == "max" or int(v) > 1 << 60 else round(int(v) / 2**30, 1)
            break
        except (OSError, ValueError):
            continue
    try:
        job["ram_total_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        pass
    d = jax.devices()[0]
    stats = d.memory_stats() or {}
    view = {"devices": [str(x) for x in jax.devices()], "kind": getattr(d, "device_kind", ""), "jax": jax.__version__,
            "jax_memory_limit_gb": round(stats.get("bytes_limit", 0) / 2**30, 2), "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")}
    measured = {}
    for dtype, name in ((jnp.float32, "f32 (TF32 allowed)"), (jnp.bfloat16, "bf16")):  # dense matrix products
        n = 8192
        x = jnp.ones((n, n), dtype)
        f = jax.jit(lambda a: a @ a)
        f(x).block_until_ready()
        t = time.perf_counter()
        for _ in range(10):
            y = f(x)
        y.block_until_ready()
        measured[f"matmul_tflops_{name}"] = round(10 * 2 * n**3 / (time.perf_counter() - t) / 1e12, 1)
    x = jnp.ones((256 * 2**20,), jnp.float32)  # 1 GiB: read + write per copy
    g = jax.jit(lambda a: a * 1.0001)
    g(x).block_until_ready()
    t = time.perf_counter()
    for _ in range(10):
        x = g(x)
    x.block_until_ready()
    measured["memory_gb_per_s"] = round(10 * 2 * x.nbytes / (time.perf_counter() - t) / 1e9)
    return {"card": card, "job": job, "jax": view, "measured": measured}


def git(path, *args):
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run(cmd, log, timeout):
    """Run a trainer; (seconds, exit code). Its output goes to `log`."""
    t = time.time()
    with open(log, "w") as f:
        try:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=ROOT).returncode
        except subprocess.TimeoutExpired:
            code = -9
    return time.time() - t, code


def speeds(log):
    """The chunk lines' steps per second, in order."""
    out = []
    for line in Path(log).read_text(errors="replace").splitlines():
        m = re.match(r"\s*([\d,]+) steps\s+([\d.]+)/s", line)
        if m:
            out.append(float(m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="shorter runs: rougher numbers")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--steps-a", type=int, default=4_000_000, help="stage A's planned steps, for the estimate")
    ap.add_argument("--steps-bcd", type=int, default=5_000_000, help="stages B, C and D's planned steps each, for the estimate")
    a = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    host = socket.gethostname().split(".")[0]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath, tpath = out / f"benchmark-{host}-{stamp}.json", out / f"benchmark-{host}-{stamp}.txt"
    scratch = Path(tempfile.mkdtemp(prefix="f1bench-"))
    t0 = time.time()
    res = {"host": host, "when": stamp, "gpu": gpu_info(), "f1sim_git": git(ROOT, "rev-parse", "HEAD"), "stages": {}, "eval": {}, "estimate": {}, "logs": {}}
    lines = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    def save():
        res["minutes"] = round((time.time() - t0) / 60, 1)
        jpath.write_text(json.dumps(res, indent=1))
        tpath.write_text("\n".join(lines) + "\n")

    say(f"checker on {host}: {res['gpu']} | f1sim {res['f1sim_git'][:10]}")
    say("\nthe GPU and what this job may use of it")
    try:
        res["gpu_specs"] = gpu_specs()
        for part, vals in res["gpu_specs"].items():
            say(f"  {part}: " + "; ".join(f"{k} {v}" for k, v in vals.items() if v not in ("", None)))
    except Exception as e:  # noqa: BLE001 - the specs are extra: the speed runs go on without them
        res["gpu_specs"] = {"error": repr(e)}
        say(f"  could not read them: {e!r}")
    save()
    steps_a = 4000 if a.quick else 8000
    steps_b = 3000 if a.quick else 6000
    common = ["--generate", "10", "--eval-tracks", "2", "--replay", "200000", "--tracks-per-set", "5", "--out", str(scratch)]

    # stage A: from scratch, learning at once
    say("\nstage A (Middle alone, 64 races of one car), learning")
    log = scratch / "a.log"
    secs, code = run([str(BIN / "f1sim-train"), *common, "--steps", str(steps_a), "--eval-every", "100000000"], log, 1200)
    sp = speeds(log)
    res["logs"]["a"] = Path(log).read_text(errors="replace")[-3000:]
    if code != 0 or len(sp) < 3:
        say(f"  failed (exit {code}): see the log tail in the json")
        save()
        return
    res["stages"]["a"] = {"steps_per_s": round(sorted(sp[2:])[len(sp[2:]) // 2], 1), "wall_s": round(secs)}
    say(f"  {res['stages']['a']['steps_per_s']:.1f} steps/s")
    save()
    init = scratch / "middle" / "stage-a" / "latest.npz"

    # stages B, C, D: from that driver, the warm start shortened so learning starts at once
    for stage, what in (("b", "Middle in 16 seats, every rule"), ("c", "B plus the team orders"), ("d", "Smooth, Aggressive, Defensive together, orders on")):
        say(f"\nstage {stage.upper()} ({what}), learning")
        log = scratch / f"{stage}.log"
        cmd = [str(BIN / "f1sim-train-b"), "--stage", stage, "--init-all", str(init), *common, "--steps", str(steps_b), "--eval-every", "100000000", "--critic-steps", "0", "--actor-ramp", "0", "--reference", "none"]
        secs, code = run(cmd, log, 1500)
        sp = speeds(log)
        res["logs"][stage] = Path(log).read_text(errors="replace")[-3000:]
        if code != 0 or len(sp) < 3:
            say(f"  failed (exit {code}): see the log tail in the json")
            save()
            continue
        txt = Path(log).read_text(errors="replace")
        m = re.search(r"eval weekend on \d+ tracks.*\(([\d,]+) steps in (\d+) s\)", txt)
        rate = re.search(r"(\d+) gradient steps? (?:per environment step|every (\d+) environment steps)", txt)
        res["stages"][stage] = {"steps_per_s": round(sorted(sp[2:])[len(sp[2:]) // 2], 1), "wall_s": round(secs), "gradient_steps": rate.group(0) if rate else None}
        say(f"  {res['stages'][stage]['steps_per_s']:.1f} steps/s ({res['stages'][stage]['gradient_steps']})")
        save()

    # the evals: the one-lap qualifying session runs to its set end whatever the cars do
    say("\nevals (untrained drivers, scaled to a trained field)")
    probe = scratch / "eval_probe.py"
    probe.write_text('''
import json, os, sys, time
import jax
from f1env import generatemap
from f1env.sensors import make_bank
from f1sim import tqc
from f1sim.drivers import env as de, evaluate, season
from f1sim.drivers.config import load
s = load("middle")
sb = dict(s.stage_b)
tracks = [generatemap(seed=900000 + k) for k in range(10)]
actors = [tqc.init(jax.random.PRNGKey(k), s.tqc).actor for k in range(2)]
out = {}
env = season.weekend_env(tracks[:5], s.preset["eval_laps"], sb)
times = {}
orig = season.run_session
def timed(env, session, *a, **k):
    t = time.time(); r = orig(env, session, *a, **k); jax.block_until_ready(r[1].laps)
    times.setdefault(session.kind, []).append((r[2], time.time() - t)); return r
season.run_session = timed
for _ in range(2):
    season.eval_weekend(env, actors[:1], s.tqc, int(sb.get("eval_seeds", 2)), names=("middle",), reference=actors[1], march_steps=s.env.march_steps)
n, t = times["one-lap"][-1]
out["weekend_steps_per_s"] = n / t
# stage A's eval: 20 races of one car, timed over a fixed number of steps
bank = make_bank(tracks)
ecfg = evaluate.eval_config(10, s.preset["eval_laps"])
st, obs = de.compiled("reset", ecfg)(bank, jax.random.PRNGKey(1), tid=jax.numpy.arange(10))
import jax.numpy as jnp
@jax.jit
def go(st, obs, key):
    def body(c, k):
        st, obs = c
        a = tqc.act(actors[0], s.tqc, obs.reshape(-1, de.OBS), k, mean=True)
        st, obs, _, _ = de.step(bank, st, a.reshape(10, 1, de.ACTIONS), k, ecfg)
        return (st, obs), None
    return jax.lax.scan(body, (st, obs), jax.random.split(key, 2000))[0]
st, obs = go(st, obs, jax.random.PRNGKey(2)); jax.block_until_ready(obs)
t = time.time(); st, obs = go(st, obs, jax.random.PRNGKey(3)); jax.block_until_ready(obs)
out["alone_steps_per_s"] = 2000 / (time.time() - t)
print(json.dumps(out))
''')
    log = scratch / "eval.log"
    secs, code = run([sys.executable, str(probe)], log, 900)
    txt = Path(log).read_text(errors="replace")
    m = re.findall(r"^\{.*\}$", txt, re.M)
    if code == 0 and m:
        e = json.loads(m[-1])
        wk, al = e["weekend_steps_per_s"] * TRAINED_FACTOR, e["alone_steps_per_s"] * TRAINED_FACTOR
        res["eval"] = {"weekend_steps_per_s_untrained": round(e["weekend_steps_per_s"]), "weekend_min_est": round(EVAL_STEPS_B / wk / 60, 1),
                       "alone_steps_per_s_untrained": round(e["alone_steps_per_s"]), "alone_min_est": round(EVAL_STEPS_A / al / 60, 1)}
        say(f"  stage B-D eval weekend (5 tracks x 2): ~{res['eval']['weekend_min_est']} min each")
        say(f"  stage A eval (alone, 10 tracks): ~{res['eval']['alone_min_est']} min each")
    else:
        res["logs"]["eval"] = txt[-3000:]
        say(f"  failed (exit {code}): see the log tail in the json")
    save()

    # the estimate: hours per stage and 24-hour jobs
    say(f"\nestimate (A {a.steps_a:,} steps, evals every 50k; B, C, D {a.steps_bcd:,} steps each, evals every 100k)")
    for stage, steps, every, ev in (("a", a.steps_a, 50_000, res["eval"].get("alone_min_est")), ("b", a.steps_bcd, 100_000, res["eval"].get("weekend_min_est")),
                                    ("c", a.steps_bcd, 100_000, res["eval"].get("weekend_min_est")), ("d", a.steps_bcd, 100_000, res["eval"].get("weekend_min_est"))):
        if stage not in res["stages"]:
            continue
        train_h = steps / res["stages"][stage]["steps_per_s"] / 3600
        eval_h = (steps // every) * (ev or 0.0) / 60
        total = train_h + eval_h
        jobs = int(-(-total // (JOB_H - 0.5)))  # half an hour a job for compiling, the warm start and saving
        res["estimate"][stage] = {"train_h": round(train_h, 1), "eval_h": round(eval_h, 1), "total_h": round(total, 1), "jobs_24h": jobs}
        say(f"  stage {stage.upper()}: {train_h:5.1f} h training + {eval_h:4.1f} h evals = {total:5.1f} h -> {jobs} job(s) of 24 h")
    say(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    save()
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"results: {jpath}\n         {tpath}")


if __name__ == "__main__":
    main()
