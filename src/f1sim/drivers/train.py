"""Stage A training (Phase 2, section 3.2), the first of the regime A-B-C-D (`stage_b`): Middle alone, one car per
race, from scratch.

    f1sim-train --tracks tracks                                   # Middle from scratch
    f1sim-train --tracks tracks --init runs/middle/stage-a/latest.npz   # carry on after a stop
    f1sim-train --generate 4 --eval-tracks 2 --steps 2000         # no pool: a small smoke run

It trains for `steps_a` environment steps, or until the eval stops improving for `patience` evals once cars finish
races (`--patience 0`: never early). Everything is written to runs/middle/stage-a/: log.jsonl (every chunk and eval),
latest.npz and best.npz, which stage B starts from. A run from scratch learns at once; one from a checkpoint
(`--init`) fills part of its replay, then trains the critics alone first (`[warm_start]`, `loop.schedule`).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")  # JAX stops at 75% of the GPU by default

import jax
import jax.numpy as jnp
import numpy as np

from f1env.sensors import make_bank

from .. import checkpoint, loop, tqc
from ..stop import StopFlag
from .. import replay as rp
from . import env as de
from . import evaluate, rewards
from .config import load
from .tracks import Rotation, eval_tracks, rotate_races


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
        "off_track_share": float(amounts[rewards.TERMS.index("off_track")] / max(tot["valid"] * 0.05, 1e-9)),
    }
    if learning.any():
        rec.update({k: float(np.asarray(m[k])[learning].mean()) for k in tqc.METRICS})
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage A: Middle alone on the track, from scratch.")
    ap.add_argument("--preset", default="laptop")
    ap.add_argument("--tracks", help="track pool directory (f1env-tracks --out ...)")
    ap.add_argument("--generate", type=int, default=0, help="no pool: generate this many training tracks")
    ap.add_argument("--eval-tracks", type=int, default=10, help="eval tracks used (the first n of the 10)")
    ap.add_argument("--init", help="carry on from this checkpoint (default: from scratch)")
    ap.add_argument("--steps", type=int, help="override steps_a (environment steps)")
    ap.add_argument("--eval-every", type=int, help="override eval_every (environment steps)")
    ap.add_argument("--patience", type=int, help="override patience (evals without a better one before stopping; 0: never early)")
    ap.add_argument("--rotate-every", type=int, help="override rotate_every (environment steps; 0: keep one set of tracks)")
    ap.add_argument("--tracks-per-set", type=int, help="override tracks (two sets are on the GPU when rotating)")
    ap.add_argument("--replay", type=int, help="override replay (transitions)")
    ap.add_argument("--critic-steps", type=int, help="override [warm_start] critic_steps (with --init; gradient steps; 0: the actor ramps up at once)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.tracks is None and not a.generate:
        ap.error("give a track pool (--tracks) or --generate N")

    s = load("middle", a.preset)
    p = s.preset
    p.update({k: v for k, v in (("tracks", a.tracks_per_set), ("replay", a.replay)) if v})
    out = Path(a.out) / s.profile / "stage-a"
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.jsonl", "a")

    def log(rec):
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()

    key = jax.random.PRNGKey(a.seed)
    k_agent, key = jax.random.split(key)
    agent = tqc.init(k_agent, s.tqc)
    if a.init:
        base, info = checkpoint.load(a.init, agent)
        agent = tqc.from_networks(s.tqc, base.actor, base.critic, base.target, base.log_alpha)
        print(f"{s.profile}: carrying on from {a.init} ({info.get('env_steps', 0):,} steps)")
    slots = p["races"] * s.env.cars
    rows = rp.rows_for(p["replay"], slots)
    replay = rp.new(rows, slots, de.OBS, de.ACTIONS, de.storage_codec())  # inputs packed
    ws = {**s.warm_start, **({} if a.critic_steps is None else {"critic_steps": a.critic_steps})}
    plan = loop.schedule(bool(a.init), rows, slots, p["updates"], s.tqc.batch, s.tqc.n_step, ws["fill"], ws["critic_steps"], ws["actor_ramp"])
    if a.init:
        print(f"warm start: the replay fills to {100 * ws['fill']:.0f}% (step {plan.learn_from:,}), then the critics learn alone for {ws['critic_steps']:,} gradient steps (to step {plan.actor_from:,}), then the actor ramps up to full over {ws['actor_ramp']:,} (to step {plan.full_from:,})")
    else:
        print(f"from scratch: learning from step {plan.learn_from:,}")
    own = s.stage_a.get("rewards", {})  # stage A's own base values (a milder crash cost), x the reward scale
    weights = np.array(s.weights)
    for term, value in own.items():
        weights[rewards.TERMS.index(term)] = float(value) * s.reward_scale
    if own:
        print("stage A rewards: " + ", ".join(f"{t} {v:g}" for t, v in own.items()) + " (the other stages: the [rewards] values)")
    w = jnp.asarray(weights)
    meta = {"reward_scale": s.reward_scale, **checkpoint.provenance(), "profile": s.profile, "preset": a.preset, "stage": "A", "eval_kind": "alone", "eval_laps": p["eval_laps"], "eval_version": evaluate.EVAL_VERSION, "init": a.init, "stage_a_rewards": own}
    print(f"{s.profile}: d={s.tqc.drop} ({s.tqc.drop * s.tqc.critics} of {s.tqc.critics * s.tqc.quantiles} atoms dropped)  H0={s.tqc.target_entropy}  gamma={s.tqc.gamma}  {p['races']} races x 1 car  replay {p['replay']:,}")

    eval_bank = make_bank(eval_tracks(a.tracks, "normal", a.eval_tracks))
    # an earlier run's best (same eval) stays the bar, so a new run's first eval can't overwrite a better checkpoint
    best = checkpoint.best_bar(out / "best.npz", stage="A", eval_kind="alone", eval_laps=p["eval_laps"], eval_version=evaluate.EVAL_VERSION)
    if best:
        print(f"best so far: {evaluate.summary(best)}")
    rot = Rotation(a.tracks, "normal", a.generate or p["tracks"], seed=a.seed, generate=bool(a.generate))
    bank = rot.bank
    key, k_reset = jax.random.split(key)
    env_state, obs = de.compiled("reset", s.env)(bank, k_reset, pick=rot.pick)
    carry = loop.Carry(env_state, obs.reshape(slots, de.OBS), agent, replay, key, jnp.zeros((), jnp.int32))
    chunk = loop.make_chunk(partial(stage_a_step, cfg=s.env, w=w), s.tqc, p["chunk"], p["updates"], 0, plan.learn_from, plan.actor_from, plan.full_from)
    print(f"tracks: {rot.describe()}; {len(eval_bank.length)} eval tracks x {p['eval_seeds']} runs")
    budget = a.steps or p["steps_a"]
    eval_every = a.eval_every or p["eval_every"]
    rotate_every = p.get("rotate_every", 0) if a.rotate_every is None else a.rotate_every
    patience = s.stage_a["patience"] if a.patience is None else a.patience
    next_eval, next_rotate = eval_every, (rotate_every if rot.rotates and rotate_every else None)
    stale, t_start, stop = 0, time.time(), StopFlag()
    while int(carry.steps) < budget and (patience == 0 or stale < patience) and not stop.asked:
        t0 = time.time()
        carry, (info, m) = chunk(carry, bank)
        jax.block_until_ready(carry.steps)
        n = int(carry.steps)
        rec = chunk_record(info, m, weights, time.time() - t0, p["chunk"], slots)
        log({"kind": "chunk", "env_steps": n, **rec})
        alpha = f"  alpha {rec['alpha']:.3f}  q {rec['q']:7.1f}  H {rec['entropy']:5.2f}" if "alpha" in rec else "  (warming up)"
        print(
            f"{n:>9,} steps  {rec['env_steps_per_s']:6.0f}/s  reward/step {rec['reward_per_step']:6.3f}  "
            f"crashes {rec['crashes']:3.0f}  laps {rec['laps']:4.0f}  off track {100 * rec['off_track_share']:4.1f}%  races {rec['races']:3.0f}{alpha}",
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
            if n <= plan.actor_from:  # the actor hasn't moved yet: the same driver, the same eval
                continue
            r = evaluate.alone(eval_bank, carry.agent.actor, s.tqc, p["eval_laps"], p["eval_seeds"], weights=weights)
            log({"kind": "eval", "env_steps": n, **r})
            info_ckpt = {**meta, "env_steps": n, "eval": r}
            checkpoint.save(out / "latest.npz", carry.agent, info_ckpt)
            improved = evaluate.better(r, best)
            if improved:
                best, stale = r, 0
                checkpoint.save(out / "best.npz", carry.agent, info_ckpt)
            elif best["finish_rate"] > 0:  # "stops improving" only counts once cars finish races
                stale += 1
            note = "  (best)" if improved else (f"  (no better: {stale}/{patience or 'no limit'})" if best["finish_rate"] > 0 else "  (no better; nothing finishes yet)")
            print(f"  eval: {evaluate.summary(r)}{note}", flush=True)
    checkpoint.save(out / "latest.npz", carry.agent, {**meta, "env_steps": int(carry.steps)})
    print(f"stage A: {int(carry.steps):,} steps in {(time.time() - t_start) / 60:.1f} min; best eval: {evaluate.summary(best) if best else 'none'}")
    log_file.close()


if __name__ == "__main__":
    main()
