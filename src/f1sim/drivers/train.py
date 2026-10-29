"""Stage A training (Phase 2, section 3.2): one profile, one car per race, on the track-width curriculum.

    f1sim-train --profile middle --tracks tracks --preset cluster                 # Middle, from scratch
    f1sim-train --profile smooth --tracks tracks --preset cluster                 # from Middle's stage A best
    f1sim-train --profile middle --generate 4 --eval-tracks 2 --preset laptop     # no pool: a small smoke run

Each width (~26 m, ~20 m, then the 12-15 m tracks) trains until the eval stops improving for `patience` evals once
cars finish races (or `steps_per_width`); the replay buffer and the networks carry on to the next width. The eval runs on the eval tracks
at the same width. Everything is written to runs/<profile>/stage-a/: log.jsonl (every chunk and eval), latest.npz,
best-<width>.npz and best.npz (the best eval on the 12-15 m tracks).

Middle trains from scratch; Aggressive, Smooth and Defensive start from Middle's best stage A networks
(`start_from` in their profile), with their own d, H0, gamma and rewards, and no random warm-up.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.sensors import make_bank

from .. import checkpoint, loop, tqc
from .. import replay as rp
from . import env as de
from . import evaluate, rewards
from .config import load
from .tracks import Rotation, eval_tracks, rotate_races, widened


def stage_a_step(bank, state, action, key, cfg: de.EnvConfig, w):
    """The driver env as the train loop sees it: slots = races (one car each), the profile's reward."""
    state, obs, out, info = de.step(bank, state, action.reshape(cfg.races, cfg.cars, de.ACTIONS), key, cfg)
    reward = out.amounts @ w
    return state, obs.reshape(-1, de.OBS), loop.Step(reward.reshape(-1), out.valid.reshape(-1), out.end.reshape(-1)), info


def chunk_record(info, m, w, elapsed: float, steps: int, slots: int) -> dict:
    tot = {k: float(np.asarray(v).sum()) for k, v in info.items() if k != "amounts"}
    amounts = np.asarray(info["amounts"]).sum(axis=0)
    valid = max(tot["valid"], 1.0)
    learning = np.asarray(m["learning"]) > 0
    rec = {
        "env_steps_per_s": steps / elapsed,
        "transitions": tot["valid"],
        "reward_per_step": float(amounts @ w / valid),
        "reward_terms": {t: float(a * wt / valid) for t, a, wt in zip(rewards.TERMS, amounts, w) if a},
        "crashes": tot["crashes"],
        "crashes_per_lap": tot["crashes"] / max(tot["laps"], 1.0),
        "laps": tot["laps"],
        "races": tot["races"],
        "finishes": tot["finishes"],
        "lap_ratio": tot["lap_ratio"] / max(tot["laps_counted"], 1.0),
        "pit_stops": tot["pit_stops"],
    }
    if learning.any():
        rec.update({k: float(np.asarray(m[k])[learning].mean()) for k in tqc.METRICS})
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage A: train a driver profile alone on the width curriculum.")
    ap.add_argument("--profile", default="middle")
    ap.add_argument("--preset", default="laptop")
    ap.add_argument("--tracks", help="track pool directory (f1env-tracks --out ...)")
    ap.add_argument("--generate", type=int, default=0, help="no pool: generate this many training tracks")
    ap.add_argument("--eval-tracks", type=int, default=10, help="eval tracks used (the first n of the 10)")
    ap.add_argument("--init", help="start from this checkpoint (default: the start_from profile's best stage A)")
    ap.add_argument("--widths", help='override the curriculum, e.g. "normal" or "26,normal"')
    ap.add_argument("--steps", type=int, help="override steps_per_width (environment steps)")
    ap.add_argument("--eval-every", type=int, help="override eval_every (environment steps)")
    ap.add_argument("--rotate-every", type=int, help="override rotate_every (environment steps; 0: keep one set of tracks)")
    ap.add_argument("--tracks-per-set", type=int, help="override tracks (two sets are on the GPU when rotating)")
    ap.add_argument("--replay", type=int, help="override replay (transitions)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.tracks is None and not a.generate:
        ap.error("give a track pool (--tracks) or --generate N")

    s = load(a.profile, a.preset)
    p, sa = s.preset, s.stage_a
    p.update({k: v for k, v in (("tracks", a.tracks_per_set), ("replay", a.replay)) if v})
    out = Path(a.out) / s.profile / "stage-a"
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.jsonl", "a")

    def log(rec):
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()

    key = jax.random.PRNGKey(a.seed)
    k_agent, key = jax.random.split(key)
    template = tqc.init(k_agent, s.tqc)
    init = a.init or (str(Path(a.out) / s.start_from / "stage-a" / "best.npz") if s.start_from else None)
    if init:
        base, info = checkpoint.load(init, template)
        agent = tqc.from_networks(s.tqc, base.actor, base.critic, base.target, base.log_alpha)
        warmup = 0
        print(f"{s.profile}: starting from {init} ({info.get('profile')}, {info.get('env_steps', 0):,} steps)")
    else:
        agent, warmup = template, int(sa["warmup"])
    slots = p["races"] * s.env.cars
    replay = rp.new(rp.rows_for(p["replay"], slots), slots, de.OBS, de.ACTIONS)
    warmup_steps = warmup // slots  # random actions
    learn_from = max(int(sa["warmup"]) // slots, s.tqc.batch // slots + s.tqc.n_step + 1)  # a started profile drives its own policy meanwhile
    w = jnp.asarray(s.weights)
    meta = {"profile": s.profile, "preset": a.preset, "stage": "A", "styles": s.styles, "drop": s.tqc.drop, "gamma": s.tqc.gamma, "target_entropy": s.tqc.target_entropy, "init": init}
    print(f"{s.profile}: d={s.tqc.drop} ({s.tqc.drop * s.tqc.critics} of {s.tqc.critics * s.tqc.quantiles} atoms dropped)  H0={s.tqc.target_entropy}  gamma={s.tqc.gamma}  {p['races']} races x 1 car  replay {p['replay']:,}")

    steps = jnp.zeros((), jnp.int32)
    widths = a.widths.split(",") if a.widths else sa["widths"]
    normal_eval = eval_tracks(a.tracks, "normal", a.eval_tracks)
    best_overall = None
    rotate_every = p.get("rotate_every", 0) if a.rotate_every is None else a.rotate_every
    for i_width, width in enumerate(widths):
        rot = Rotation(a.tracks, width, a.generate or p["tracks"], seed=a.seed + i_width, generate=bool(a.generate))
        # the eval at a wide width: the eval tracks' copies that fit, each run more times so every width's eval has as
        # many races; if none fit, the normal eval tracks
        ev_tracks = widened(normal_eval, width)
        if not ev_tracks:
            print(f"width {width}: no eval track fits at this width; evaluating on the normal ones")
            ev_tracks = normal_eval
        eval_seeds = -(-p["eval_seeds"] * len(normal_eval) // len(ev_tracks))
        bank, eval_bank = rot.bank, make_bank(ev_tracks)
        key, k_reset = jax.random.split(key)
        env_state, obs = de.compiled("reset", s.env)(bank, k_reset, pick=rot.pick)
        carry = loop.Carry(env_state, obs.reshape(slots, de.OBS), agent, replay, key, steps)
        chunk = loop.make_chunk(partial(stage_a_step, cfg=s.env, w=w), s.tqc, p["chunk"], p["updates"], warmup_steps, learn_from)
        print(f"width {width}: {rot.describe()}, {len(ev_tracks)} eval tracks x {eval_seeds} runs")

        budget = a.steps or p["steps_per_width"]
        start = int(carry.steps)
        eval_every = a.eval_every or p["eval_every"]
        next_eval = start + eval_every
        next_rotate = start + rotate_every if rot.rotates and rotate_every else None
        best, stale = None, 0
        t_width = time.time()
        while int(carry.steps) - start < budget and stale < sa["patience"]:
            t0 = time.time()
            carry, (info, m) = chunk(carry, bank)
            jax.block_until_ready(carry.steps)
            n = int(carry.steps)
            rec = chunk_record(info, m, s.weights, time.time() - t0, p["chunk"], slots)
            log({"kind": "chunk", "width": width, "env_steps": n, **rec})
            alpha = f"  alpha {rec['alpha']:.3f}  q {rec['q']:7.1f}  H {rec['entropy']:5.2f}" if "alpha" in rec else "  (warming up)"
            print(
                f"[{width:>6}] {n:>9,} steps  {rec['env_steps_per_s']:6.0f}/s  reward/step {rec['reward_per_step']:6.3f}  "
                f"crashes {rec['crashes']:3.0f}  laps {rec['laps']:4.0f}  races {rec['races']:3.0f}{alpha}",
                flush=True,
            )
            if next_rotate and n >= next_rotate:  # a fresh set of training tracks
                next_rotate += rotate_every
                bank, env_state, obs, hit = rotate_races(rot, carry.env, carry.obs.reshape(s.env.races, s.env.cars, de.OBS), jax.random.fold_in(carry.key, n), s.env)
                replay_cut = rp.cut_last(carry.replay, jnp.asarray(np.repeat(hit, s.env.cars)))
                carry = carry._replace(env=env_state, obs=obs.reshape(slots, de.OBS), replay=replay_cut)
                print(f"  tracks: {rot.describe()}; {int(hit.sum())} races restarted", flush=True)
            if n >= next_eval:
                next_eval += eval_every
                r = evaluate.alone(eval_bank, carry.agent.actor, s.tqc, p["eval_laps"], eval_seeds)
                log({"kind": "eval", "width": width, "env_steps": n, **r})
                info_ckpt = {**meta, "width": width, "env_steps": n, "eval": r}
                checkpoint.save(out / "latest.npz", carry.agent, info_ckpt)
                improved = evaluate.better(r, best)
                if improved:
                    best, stale = r, 0
                    checkpoint.save(out / f"best-{width}.npz", carry.agent, info_ckpt)
                    if width == "normal" and evaluate.better(r, best_overall):
                        best_overall = r
                        checkpoint.save(out / "best.npz", carry.agent, info_ckpt)
                elif best["finish_rate"] > 0:  # "stops improving" only counts once cars finish races
                    stale += 1
                note = "  (best)" if improved else (f"  (no better: {stale}/{sa['patience']})" if best["finish_rate"] > 0 else "  (no better; nothing finishes yet)")
                print(f"  eval at {width}: {evaluate.summary(r)}{note}", flush=True)
        agent, replay, key, steps = carry.agent, carry.replay, carry.key, carry.steps
        print(f"width {width} done: {int(steps) - start:,} steps in {(time.time() - t_width) / 60:.1f} min; best eval: {evaluate.summary(best) if best else 'none'}")
    checkpoint.save(out / "latest.npz", agent, {**meta, "env_steps": int(steps)})
    log_file.close()


if __name__ == "__main__":
    main()
