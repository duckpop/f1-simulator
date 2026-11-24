"""Stages B, C and D (Phase 2, section 3.2) on the full grid. The training regime (the user's, 2026-11-24):

- A: Middle alone on the track, from scratch (`train.py`, f1sim-train).
- B: Middle in a field with every rule (the safety car, flags, pit stops), from its A best (`--stage b`).
- C: Middle obeying a random principal's team orders (pace push/save, hold position, let the team-mate past,
  defend, ERS deploy/save; `env.random_orders`, the order reward terms), from its B best (`--stage c`, `[stage_c]`).
- D: the personalities, each from Middle's C best, orders still on (`--stage d`, `[stage_d]`; default Smooth,
  Aggressive and Defensive). Until D only Middle learns.

    f1sim-train-b --stage b --tracks tracks       # Middle from runs/middle/stage-a/best.npz
    f1sim-train-b --stage c --tracks tracks       # Middle from runs/middle/stage-b/best.npz
    f1sim-train-b --stage d --tracks tracks       # the personalities from runs/middle/stage-c/best.npz
    f1sim-train-b --stage b --tracks tracks --resume                  # carry on after a stop
    f1sim-train-b --generate 4 --eval-tracks 2 --init-all runs/middle/stage-a/best.npz --steps 2000   # smoke run

- 22 cars per race on the 22 m tracks, with every rule f1env has: overtaking, dirty air, slipstream, DRS, contact,
  the safety car, blue flags. Seats are fixed by car number (the grid is random every race): the learners' seats
  (`per_profile`, split between the profiles; `--per-profile` overrides), the rest frozen snapshots of earlier drivers.
- Each profile's target entropy is `entropy_shift` lower than in stage A, reached linearly over `entropy_ramp` steps;
  TQC tunes alpha towards it. The actor takes `actor_lr` (lower than the critics' `lr`).
- Each profile has its own networks, optimisers and replay buffer; each car's steps go to its profile's buffer with
  its profile's reward weights.
- The snapshot pool starts with the starting drivers (or `pool_seed`'s checkpoints, oldest first). Every slot belongs
  to a personality, and a profile's new bests only replace its own slots, the oldest first (`snapshot_on = "best"`;
  "none": the pool stays as it starts). Each snapshot drives an equal share of the frozen cars, shuffled every
  session, with its mean action.
- 20% of sessions are short qualifying sessions (`quali_share`). Each session has 6, 12, 18 or 22 cars
  (`field_sizes`), and half the races (`spread_share`) start with the cars spread around the lap at speed.
- The eval: a full weekend on every eval track (a one-lap qualifying sets the grid, then a race of `eval_laps` laps; in C and D with
  the random principal's orders for every car), the profiles in the even cars and a reference field in the odd ones:
  the stage's starting Middle, frozen (runs/<stage>-reference.npz), so a best means better than where the stage
  started. Each profile keeps its best checkpoint in runs/<profile>/<stage>/ (`evaluate.better_ordered`, its
  `best_order`).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")  # JAX stops at 75% of the GPU by default
# CUDA's own pool allocator, not XLA's BFC: b4b (2026-11-12, batch 512) died at its 16th track swap, BFC unable to find
# one 204.5 MiB block (the bank's edge grid) in memory split into gaps; BFC's message suggests the async allocator
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

import jax
import jax.numpy as jnp
import numpy as np

from .. import checkpoint, loop, nets, revive, tqc
from ..stop import StopFlag
from .. import replay as rp
from . import env as de
from . import evaluate, rewards, season
from .config import PROFILES, load
from .tracks import Rotation, eval_tracks, rotate_races

CARS = 22


def seat_layout(cars: int = CARS, per_profile: int = 4, profiles: int = len(PROFILES)) -> np.ndarray:
    """(C,) the profile each car number drives, -1 for a frozen snapshot. The learners come first, the profiles in
    turn, so team-mates (cars 2t, 2t + 1) are different profiles."""
    n = per_profile * profiles
    if n > cars:
        raise ValueError(f"{per_profile} learners per profile is more than {cars} cars")
    seats = np.full(cars, -1)
    seats[:n] = np.arange(n) % profiles
    return seats


# The training regime (the user's, 2026-11-24): A, Middle alone on the track (`train.py`); then, here, B, C and D.
class Stage(NamedTuple):
    dir: str  # runs/<profile>/<dir>/, its log runs/<dir>-log.jsonl
    label: str
    before: str  # the stage whose Middle best each profile starts from (D: every personality from Middle's)
    sections: tuple  # the config sections merged, in order ([stage_b] first)
    profiles: str  # who learns by default
    middle_only: bool  # only Middle may learn
    orders: bool  # the random principal's team orders (`env.random_orders`)


STAGES = {
    "b": Stage("stage-b", "B", "stage-a", ("stage_b",), "middle", True, False),  # a field, every rule, pit stops
    "c": Stage("stage-c", "C", "stage-b", ("stage_b", "stage_c"), "middle", True, True),  # team orders
    "d": Stage("stage-d", "D", "stage-c", ("stage_b", "stage_c", "stage_d"), "smooth,aggressive,defensive", False, True),  # the personalities
}


def init_path(profile: str, out, inits: dict, resume: bool = False, init_all: str | None = None, stage: str = "b") -> str:
    """The checkpoint a profile starts from: its own --init, else this stage's latest.npz (--resume), --init-all, or
    Middle's best from the stage before (`Stage.before`)."""
    if profile in inits:
        return inits[profile]
    if resume:
        return str(Path(out) / profile / STAGES[stage].dir / "latest.npz")
    return init_all or str(Path(out) / "middle" / STAGES[stage].before / "best.npz")


def stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


class League(NamedTuple):
    env: de.EnvState
    obs: jax.Array  # (R, C, obs) the next step's inputs
    agents: tuple  # a TQCState per profile
    replays: tuple  # a Replay per profile
    pool: dict  # K snapshot actors, stacked
    pool_of: jax.Array  # (R, frozen) the snapshot each frozen car drives this session
    key: jax.Array
    steps: jax.Array


def stored_seats(seats: np.ndarray, profile: int, share: float = 1.0) -> np.ndarray:
    """The car numbers of `profile` whose steps go to its replay: the first `share` of its seats (at least one). The
    grid is random every race, so which seats doesn't matter; storing fewer makes the same buffer cover more time."""
    idx = np.flatnonzero(seats == profile)
    return idx[: max(1, int(round(len(idx) * share)))]


# command-line options that override a stage setting: option -> (setting, conversion). "given" ones win whenever
# passed; "set" ones only when not empty or 0 (as before 2026-11-24: --steps 0 means the config's)
CLI_GIVEN = {"actor_lr": ("actor_lr", float), "disagree": ("disagree", float), "entropy_ramp": ("entropy_ramp", int), "actor_average": ("actor_average", int),
             "critic_steps": ("critic_steps", int), "actor_ramp": ("actor_ramp", int)}
CLI_SET = {"steps": ("steps", int), "eval_every": ("eval_every", int), "snapshot_on": ("snapshot_on", str),
           "reference": ("eval_reference", str), "pool_seed": ("pool_seed", lambda v: v.split(",")), "start_tyres": ("start_tyres", lambda v: [float(x) for x in v.split(",")])}


def cli_overrides(a) -> dict:
    """The stage settings the command line overrides (`CLI_GIVEN`, `CLI_SET`), under the settings' names."""
    out = {key: conv(getattr(a, opt)) for opt, (key, conv) in CLI_GIVEN.items() if getattr(a, opt) is not None}
    out.update({key: conv(getattr(a, opt)) for opt, (key, conv) in CLI_SET.items() if getattr(a, opt)})
    return out


def make_chunk(cfgs, env_cfg: de.EnvConfig, seats: np.ndarray, weights: np.ndarray, steps: int, updates: int, plan: loop.Schedule, stored=None, stride: int = 1, entropy_from=None, entropy_ramp: int = 0):
    """cfgs: a TQCConfig per profile; weights (profiles, terms). One jitted call runs `steps` environment steps, every
    `stride`-th followed by `updates` gradient steps for every profile. stored: per profile, the car numbers whose steps go to its
    replay (default: all of its seats); plan: the warm start (`loop.Schedule`: the critics learn alone from
    learn_from, the actors and temperatures from actor_from, at full size from full_from). entropy_from: per profile,
    the target entropy the run starts at, moved linearly to its cfg's over `entropy_ramp` environment steps from
    actor_from (None or 0: cfg's from the start)."""
    P, (R, C) = len(cfgs), (env_cfg.races, env_cfg.cars)
    idx = [np.flatnonzero(seats == p) for p in range(P)]
    stored = idx if stored is None else [np.asarray(x) for x in stored]
    frozen = np.flatnonzero(seats < 0)
    group = np.where(seats < 0, P, seats)  # frozen cars are group P in the stats
    onehot = jnp.asarray(np.eye(P + 1)[group], jnp.float32)  # (C, P + 1)
    w_car = jnp.asarray(np.where(seats[:, None] < 0, 0.0, weights[np.maximum(seats, 0)]), jnp.float32)  # (C, terms)
    enc = cfgs[0].encoder

    def body(bank, c: League, _):
        keys = jax.random.split(c.key, 3 + 2 * P)
        key, k_env, k_pool = keys[0], keys[1], keys[2]
        obs = c.obs.astype(jnp.float32)
        actions = jnp.zeros((R, C, de.ACTIONS))
        for p in range(P):
            a = tqc.act(c.agents[p].actor, cfgs[p], obs[:, idx[p]].reshape(-1, de.OBS), keys[3 + p]).reshape(R, len(idx[p]), de.ACTIONS)
            actions = actions.at[:, idx[p]].set(a)
        if frozen.size:
            actions = actions.at[:, frozen].set(pool_actions(c.pool, c.pool_of, obs[:, frozen], enc))

        env, nxt, out, info = de.step(bank, c.env, actions, k_env, env_cfg)
        reward = jnp.sum(out.amounts * w_car[None], axis=-1)
        agents, replays, ms = [], [], []
        on, actor_on = (c.steps >= plan.learn_from) & (c.steps % stride == 0), plan.actor_share(c.steps)  # learn on every `stride`-th step
        for p in range(P):
            sl = stored[p]
            flat = lambda x: x[:, sl].reshape((-1,) + x.shape[2:])
            buf = rp.add(c.replays[p], flat(c.obs), flat(actions), flat(reward), flat(out.valid), flat(out.end))
            h0 = None
            if entropy_from is not None and entropy_ramp > 0:  # the target on its way down to cfg's
                done = jnp.clip((c.steps - plan.actor_from) / entropy_ramp, 0.0, 1.0)
                h0 = entropy_from[p] + done * (cfgs[p].target_entropy - entropy_from[p])
            agent, m = loop.maybe_learn(on, c.agents[p], cfgs[p], buf, keys[3 + P + p], updates, actor_on, h0)
            agents.append(agent)
            replays.append(buf)
            ms.append(m)
        restarted = env.tick == 0
        pool_size = jax.tree.leaves(c.pool)[0].shape[0]
        pool_of = jnp.where(restarted[:, None], assign_pool(k_pool, R, c.pool_of.shape[1], pool_size), c.pool_of)
        stats = {
            "amounts": jnp.einsum("rck,cg->gk", out.amounts, onehot),  # (groups, terms)
            "valid": jnp.einsum("rc,cg->g", out.valid.astype(jnp.float32), onehot),
            "laps": jnp.einsum("rc,cg->g", out.lap_done.astype(jnp.float32), onehot),
            "crashes": jnp.einsum("rc,cg->g", out.crashed.astype(jnp.float32), onehot),
            "hit": jnp.einsum("rc,cg->g", out.hit.astype(jnp.float32), onehot),
            "warnings": jnp.einsum("rck,cg->gk", out.warnings, onehot),  # (groups, kinds)
            "black_by": jnp.einsum("rck,cg->gk", out.black_by, onehot),
            "reward": jnp.einsum("rc,cg->g", reward, onehot),
            "races": info["races"],
        }
        metrics = {k: jnp.stack([m[k] for m in ms]) for k in tqc.METRICS}
        metrics["learning"] = on.astype(jnp.float32)
        metrics["actor"] = on.astype(jnp.float32) * actor_on
        return League(env, nxt, tuple(agents), tuple(replays), c.pool, pool_of, key, c.steps + 1), (stats, metrics)

    @partial(jax.jit, donate_argnums=0)
    def chunk(league: League, bank):
        return jax.lax.scan(partial(body, bank), league, None, length=steps)

    return chunk


def pool_actions(pool, pool_of, obs, enc):
    """(R, F, actions) each frozen car's mean action from its own snapshot. Each snapshot runs only on the cars it
    drives (at most ceil(F / K) per race, as `assign_pool` shares them out), not on every frozen car: K times less work
    than running all K on all F (the same actions, up to float rounding; since 2026-11-24)."""
    R, F = pool_of.shape
    K = jax.tree.leaves(pool)[0].shape[0]
    G = -(-F // K)
    mine = pool_of[..., None] == jnp.arange(K)  # (R, F, K)
    rank = jnp.sum((jnp.cumsum(mine, axis=1) - 1) * mine, axis=-1)  # (R, F) each car's place among its snapshot's cars
    races = jnp.arange(R)[:, None]
    seat = jnp.zeros((R, K, G), jnp.int32).at[races, pool_of, rank].set(jnp.broadcast_to(jnp.arange(F), (R, F)))  # unused places: car 0
    grouped = jnp.take_along_axis(obs[:, None], seat[..., None], axis=2)  # (R, K, G, inputs)
    out = jax.vmap(lambda prm, x: nets.actor_mean(prm, enc, x), in_axes=(0, 1), out_axes=1)(pool, grouped)  # (R, K, G, actions)
    return out[races, pool_of, rank]


def assign_pool(key, races: int, frozen: int, size: int) -> jax.Array:
    """(races, frozen) the snapshot each frozen car drives: every snapshot an equal share of the cars (the first
    `frozen % size` one more), in a random order per race, so a smaller field still gets a random mix."""
    shares = jnp.arange(frozen) % size
    return jax.vmap(lambda k: jax.random.permutation(k, shares))(jax.random.split(key, races)).astype(jnp.int32)


def trim_heap() -> None:
    """Hand the heap's freed memory back to the system (glibc's malloc_trim). Each track rotation builds the next bank
    in host memory and frees it; glibc kept the freed pages, so the run grew ~60-80 MB a rotation (middle-sc2:
    7.8 GB at 570k, 8.6 GB at 1.3M, plus a ~0.5-0.7 GB peak at every rotation, against a 10 GB limit). With a trim
    after every rotation it stayed flat over 10 (runs/mem_probe_trim.py, 2026-11-20). No-op off glibc."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def oldest_slot(owners: list, ages: list, profile: str):
    """The pool slot a new snapshot of `profile` replaces: the oldest of the slots that profile owns, None if it owns
    none. A profile's snapshots only ever replace its own, so the other personalities' frozen cars never change."""
    own = [k for k, o in enumerate(owners) if o == profile]
    return min(own, key=lambda k: ages[k]) if own else None


def push_snapshot(league: League, slot: int, actor) -> League:
    """Copy an actor into pool slot `slot`."""
    pool = jax.tree.map(lambda s, a: s.at[slot].set(a), league.pool, actor)
    return league._replace(pool=pool)


@jax.jit
def average_in(avg, actor, share):
    """The averaged actor moved `share` of the way to the current one (Polyak averaging of the weights)."""
    return jax.tree.map(lambda m, x: m + share * (x - m), avg, actor)


def updates_for(sb: dict, default: int, stored_per_step: int, batch: int) -> int:
    """Gradient steps per environment step: enough for each stored step to be sampled `reuse` times on average
    (reuse x stored steps per environment step / batch), or the preset's `updates` without a reuse."""
    reuse = sb.get("reuse")
    return max(1, round(float(reuse) * stored_per_step / batch)) if reuse else int(default)


def learn_every(sb: dict, stored_per_step: int, batch: int) -> int:
    """Environment steps per round of gradient steps: 1, or more when a `reuse` asks for under one gradient step per
    environment step (each stored step sampled `reuse` times: one gradient step every batch / (reuse x stored))."""
    reuse = sb.get("reuse")
    want = float(reuse) * stored_per_step / batch if reuse else 1.0
    return max(1, round(1.0 / want)) if want < 1.0 else 1


def chunk_record(stats, metrics, profiles, elapsed: float, steps: int, seats_per_group=None) -> dict:
    """seats_per_group: (profiles + 1,) cars in each group over all races (the frozen ones last), for the share alive."""
    amounts = np.asarray(stats["amounts"]).sum(axis=0)  # (groups, terms)
    valid = np.asarray(stats["valid"]).sum(axis=0)
    laps = np.asarray(stats["laps"]).sum(axis=0)
    crashes = np.asarray(stats["crashes"]).sum(axis=0)
    hit = np.asarray(stats["hit"]).sum(axis=0)
    warned = np.asarray(stats["warnings"]).sum(axis=0) if "warnings" in stats else None  # (groups, kinds)
    black_by = np.asarray(stats["black_by"]).sum(axis=0) if "black_by" in stats else None
    reward = np.asarray(stats["reward"]).sum(axis=0)
    learning = np.asarray(metrics["learning"]) > 0
    t = {name: i for i, name in enumerate(rewards.TERMS)}
    rec = {"env_steps_per_s": steps / elapsed, "races": float(np.asarray(stats["races"]).sum()), "profiles": {}}
    for g, name in enumerate([*profiles, "frozen"]):
        v = max(valid[g], 1.0)
        r = {
            "transitions": float(valid[g]),
            "laps": float(laps[g]),
            "crashes": float(crashes[g]),
            "crashes_per_lap": float(crashes[g] / max(laps[g], 1.0)),
            "hit": float(hit[g]),
            "black_flags": float(amounts[g, t["black_flag"]]),
            "overtakes": float(amounts[g, t["overtake"]]),
            "places_lost": float(amounts[g, t["place_lost"]]),
            "holding_s": float(amounts[g, t["holding"]]),
            "finishes": float(amounts[g, t["finish"]]),
        }
        if warned is not None:
            r["warnings_by"] = {k: float(warned[g, i]) for i, k in enumerate(de.WARNING_KINDS)}
            r["black_flags_by"] = {k: float(black_by[g, i]) for i, k in enumerate(de.WARNING_KINDS)}
        if seats_per_group is not None:
            r["alive"] = float(valid[g] / max(seats_per_group[g] * steps, 1))
        if g < len(profiles):
            r["reward_per_step"] = float(reward[g] / v)
            if learning.any():
                r.update({k: float(np.asarray(metrics[k])[learning, g].mean()) for k in tqc.METRICS})
                r["critics_only"] = bool(not (np.asarray(metrics["actor"])[learning] > 0).any())
                r["actor_share"] = float(np.asarray(metrics["actor"])[learning].mean())  # the actor's ramp, 0..1
        rec["profiles"][name] = r
    return rec


def kinds_text(r: dict) -> str:
    """Black flags by the kind of the 4th warning and every warning by kind, as `[s1 p0 b0 m2 x0 c0 f0 | w s9 p1 b3 m4
    x1 c2 f1]` (speeding, passing, blue flag, not closing on the safety car (under the minimum until 2026-11-19),
    crossing the pit exit's blend line, too close and too far back in the safety car's queue); empty without them (a
    log from before 2026-11-12 has no x, one from before 2026-11-19 no c or f)."""
    if "warnings_by" not in r:
        return ""
    letter = {"slow": "m", "blend": "x"}
    short = lambda d: " ".join(f"{letter.get(k, k[0])}{v:.0f}" for k, v in d.items())
    return f" [{short(r['black_flags_by'])} | w {short(r['warnings_by'])}]"


def line_part(name: str, r: dict) -> str:
    """One group's part of the log line: the share of its seats alive, crashes (per lap; how many it was hit in), black
    flags (by the 4th warning's kind, then all warnings by kind: `kinds_text`), laps, overtakes; for a learning
    profile its reward per step and the critics' Q, alpha and entropy."""
    out = f"{name[:3]} alive {100 * r.get('alive', float('nan')):3.0f}% cr {r['crashes']:3.0f} ({r['crashes_per_lap']:.2f}/lap, hit {r.get('hit', 0):.0f}) bf {r.get('black_flags', 0):.0f}{kinds_text(r)} lap {r['laps']:4.0f} ov {r['overtakes']:3.0f}"
    if "reward_per_step" in r:
        out += f" r {r['reward_per_step']:6.3f}"
    if "q" in r:
        out += f" q {r['q']:6.1f} sp {r.get('spread', 0.0):5.2f} α {r['alpha']:.4f} H {r['entropy']:5.2f}" + (" (critics only)" if r.get("critics_only") else f" (actor ramp {100 * r['actor_share']:.0f}%)" if r.get("actor_share", 1.0) < 0.999 else "")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stages B, C and D on full grids (stage A: f1sim-train).")
    ap.add_argument("--stage", choices=sorted(STAGES), default="b", help="b: Middle in a field with every rule; c: Middle obeying team orders; d: the personalities from Middle's C best, orders on")
    ap.add_argument("--preset", default="laptop")
    ap.add_argument("--tracks", help="track pool directory (f1env-tracks --out ...)")
    ap.add_argument("--generate", type=int, default=0, help="no pool: generate this many training tracks")
    ap.add_argument("--eval-tracks", type=int, help="the first n eval tracks (default: [stage_b] eval_tracks, the five picked by layout)")
    ap.add_argument("--init-all", help="start every profile from this checkpoint (default: Middle's best from the stage before)")
    ap.add_argument("--resume", action="store_true", help="continue every profile from its own latest.npz in this stage (after a stop)")
    ap.add_argument("--init", action="append", default=[], metavar="PROFILE=PATH", help="start this profile from this checkpoint (repeatable; over --init-all and --resume)")
    ap.add_argument("--steps", type=int, help="override steps_b (environment steps)")
    ap.add_argument("--eval-every", type=int, help="override eval_every_b (environment steps)")
    ap.add_argument("--critic-steps", type=int, help="override [warm_start] critic_steps (gradient steps; 0: the actors ramp up at once)")
    ap.add_argument("--actor-ramp", type=int, help="override [warm_start] actor_ramp (gradient steps; 0: the actors at full at once)")
    ap.add_argument("--snapshot-on", choices=("best", "none"), help="override snapshot_on: each new best eval's driver into the pool, or none (the pool stays as it starts)")
    ap.add_argument("--pool-seed", help="override pool_seed: the frozen pool's starting checkpoints, comma-separated, oldest first")
    ap.add_argument("--entropy-ramp", type=int, help="override entropy_ramp (environment steps the target entropy takes to reach its stage B value; 0: there from the start, for a run continued after the ramp)")
    ap.add_argument("--rotate-every", type=int, help="override rotate_every (environment steps; 0: keep one set of tracks)")
    ap.add_argument("--tracks-per-set", type=int, help="override tracks (two sets are on the GPU when rotating)")
    ap.add_argument("--replay", type=int, help="override replay_b (transitions per profile)")
    ap.add_argument("--profiles", help="the profiles that learn, comma-separated (B and C: middle only; D default: smooth, aggressive, defensive)")
    ap.add_argument("--per-profile", type=int, help="learning cars per profile (default: 16 learning seats split between the profiles)")
    ap.add_argument("--reference", help='the eval\'s fixed reference driver (default: the stage\'s starting Middle, frozen; "none": no reference field)')
    ap.add_argument("--actor-lr", type=float, help="override actor_lr (the actor's own learning rate)")
    ap.add_argument("--actor-average", type=int, help="override actor_average (half-life in steps; 0: no averaged copy, the evals, bests and snapshots take the training actor)")
    ap.add_argument("--disagree", type=float, help="override disagree: the actor rates actions at the critics' mean less this x their spread (0: off)")
    ap.add_argument("--start-tyres", help="override start_tyres: soft,medium,hard shares of the dry starting compound in training races (evals: mediums)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.tracks is None and not a.generate:
        ap.error("give a track pool (--tracks) or --generate N")
    st = STAGES[a.stage]
    names = tuple((a.profiles or st.profiles).split(","))
    if set(names) - set(PROFILES) or len(set(names)) != len(names):
        ap.error(f"--profiles takes some of {', '.join(PROFILES)}, each once: {a.profiles}")
    if st.middle_only and names != ("middle",):
        ap.error(f"stage {a.stage.upper()} trains Middle alone (the personalities branch off in stage D): {a.profiles}")
    inits = dict(x.split("=", 1) for x in a.init if "=" in x)
    if len(inits) != len(a.init) or set(inits) - set(names):
        ap.error(f"--init takes PROFILE=PATH, once per profile ({', '.join(names)}): {' '.join(a.init)}")

    settings = [load(p, a.preset) for p in names]
    s0 = settings[0]
    p = s0.preset
    sub, label = st.dir, st.label
    sb = {k: v for section in st.sections for k, v in getattr(s0, section).items()}  # [stage_b], then the stage's own
    sb.update(cli_overrides(a))  # the command line over the config: everything below reads `sb`
    p.update({k: v for k, v in (("tracks", a.tracks_per_set), ("replay_b", a.replay or (sb.get("replay") if len(names) == 1 else None))) if v})  # the stage's replay is one profile's
    # this stage's learning settings: lr (actor, critics and alpha), the actor's own lower actor_lr, the critics'
    # quantile Huber cut-off kappa, the disagreement penalty
    tuned = {k: float(sb[k]) for k in ("lr", "actor_lr", "kappa", "disagree") if sb.get(k) is not None}
    entropy_ramp = int(sb.get("entropy_ramp", 0))
    cfgs = [s.tqc._replace(target_entropy=s.tqc.target_entropy + sb.get("entropy_shift", 0.0), **tuned) for s in settings]  # less noise
    per = a.per_profile or sb["per_profile"] * len(PROFILES) // len(names)
    seats = seat_layout(CARS, per, len(names))
    # a race restarts once its learners are all out (frozen cars alone don't keep it going)
    stored = [stored_seats(seats, i, float(sb.get("store_share", 1.0))) for i in range(len(names))]
    kept = np.zeros(CARS, bool)
    kept[np.concatenate(stored)] = True  # in a small field the stored seats race first, so the replay keeps filling
    env_cfg = s0.env._replace(
        races=p["races_b"], cars=CARS, quali_share=sb["quali_share"], quali_s=sb["quali_s"], learners=tuple(bool(x) for x in seats >= 0),
        field_sizes=tuple(int(n) for n in sb.get("field_sizes", ())), keep=tuple(bool(x) for x in kept),
        spread_share=float(sb.get("spread_share", 0.0)), spread_speed=tuple(float(x) for x in sb.get("spread_speed", (0.6, 0.9))),
        start_tyres=tuple(float(x) for x in sb.get("start_tyres", ())),
        **{k: type(getattr(s0.env, k))(sb[k]) for k in ("min_laps", "length_skew", "two_compound_km", "learners_first") if k in sb},
        random_orders=st.orders, **({"order_every": tuple(float(x) for x in sb["order_every"])} if "order_every" in sb else {}),
        **({"order_rest": float(sb["order_rest"])} if "order_rest" in sb else {}),
    )
    env_cfg = env_cfg._replace(race=de.stage_race(env_cfg.race, sb))  # the stuck times and the safety car band, as in the evals
    R = env_cfg.races

    key = jax.random.PRNGKey(a.seed)
    agents, replays, done = [], [], 0
    for i, s in enumerate(settings):
        template = tqc.init(jax.random.PRNGKey(0), cfgs[i])
        init = init_path(s.profile, a.out, inits, a.resume, a.init_all, a.stage)
        base, info = checkpoint.load(init, template)
        done += info.get("env_steps", 0)
        agents.append(base)  # its networks and optimisers carry on
        slots = R * len(stored[i])
        rows = rp.rows_for(p["replay_b"], slots)
        replays.append(rp.new(rows, slots, de.OBS, de.ACTIONS, de.storage_codec()))
        print(f"{s.profile:>10}: from {init} ({info.get('profile')} stage {info.get('stage', '?')}, {info.get('env_steps', 0):,} steps)  d={s.tqc.drop} H0={cfgs[i].target_entropy} gamma={s.tqc.gamma} lr={cfgs[i].lr:g} kappa={cfgs[i].kappa:g}")
        gb = rows * slots * rp.bytes_per_step(de.OBS, de.ACTIONS, de.storage_codec()) / 1e9
        print(f"{'':>10}  replay: {len(stored[i])} of {int((seats == i).sum())} seats per race stored, {rows * slots:,} steps in {rows:,} rows ({rows * de.STEP_S / 60:.0f} min of racing, {gb:.2f} GB packed)")
    K = sb["snapshots"]
    seed_paths = list(sb.get("pool_seed", []))
    if seed_paths:  # given checkpoints, oldest first, repeated in order to fill the pool (the first ones go out first)
        template = tqc.init(jax.random.PRNGKey(0), s0.tqc)
        # "profile=path" names the slot's personality; a bare path takes the checkpoint's own profile
        named = [sp.split("=", 1) if "=" in sp else (None, sp) for sp in seed_paths]
        loaded = [checkpoint.load(path, template) for _, path in named]
        seeded = [ag.actor for ag, _ in loaded]
        seed_owner = [name or info.get("profile", "?") for (name, _), (_, info) in zip(named, loaded)]
        pick_k = [k * len(seeded) // K for k in range(K)]
        pool = stack([seeded[j] for j in pick_k])
        owners = [seed_owner[j] for j in pick_k]
        print(f"snapshot pool from {len(seed_paths)} checkpoints: " + ", ".join(f"{owners[k]}:{Path(named[j][1]).stem}" for k, j in enumerate(pick_k)))
    else:
        pool = stack([agents[k % len(agents)].actor for k in range(K)])
        owners = [settings[k % len(agents)].profile for k in range(K)]
    ages = list(range(K))  # the order the slots were filled in: the seed's first ones are the oldest
    n_frozen = int((seats < 0).sum())
    print(f"stage {label}: every rule (rain, the safety car, red, blue and black flags)" + ("; a random principal gives the learners team orders" if st.orders else ""))
    print(f"{R} races x {CARS} cars: {per} per profile learning, {n_frozen} frozen ({K} snapshots); {100 * sb['quali_share']:.0f}% qualifying")
    if env_cfg.field_sizes or env_cfg.spread_share:
        print(f"fields of {', '.join(map(str, env_cfg.field_sizes or (CARS,)))} cars; {100 * env_cfg.spread_share:.0f}% of races start spread around the lap at {env_cfg.spread_speed[0]:g}-{env_cfg.spread_speed[1]:g} x the reference pace")

    # the starting checkpoints' steps in the seed: a run continued from another takes the pool in a new order, not
    # the same first sets again
    rot = Rotation(a.tracks, "normal", a.generate or p["tracks"], seed=a.seed + done, generate=bool(a.generate))
    bank = rot.bank
    pick = None if a.eval_tracks else sb.get("eval_tracks")  # the eval tracks by number (else the first --eval-tracks, or all 10)
    eval_env = season.weekend_env(eval_tracks(a.tracks, "normal", a.eval_tracks or 10, pick), p["eval_laps"], sb)  # one-lap qualifying, then the race (with the orders in C and D)
    key, k_reset, k_pool = jax.random.split(key, 3)
    env_state, obs = de.compiled("reset", env_cfg)(bank, k_reset, pick=rot.pick)
    league = League(env_state, obs, tuple(agents), tuple(replays), pool, assign_pool(k_pool, R, n_frozen, K), key, jnp.zeros((), jnp.int32))
    updates = updates_for(sb, p["updates"], R * len(stored[0]), s0.tqc.batch)
    every = learn_every(sb, R * len(stored[0]), s0.tqc.batch)
    rate = f"{updates} gradient steps per environment step" if every == 1 else f"{updates} gradient step every {every} environment steps"
    print(f"{rate}: each stored step is sampled ~{updates * s0.tqc.batch / (every * R * len(stored[0])):.1f} times")
    # stages B-D always start from checkpoints (Middle's best from the stage before, or their own latest)
    ws = {**s0.warm_start, **{k: sb[k] for k in ("critic_steps", "actor_ramp") if k in sb}}  # a stage's own (or the flags)
    rows_b = min(int(r.reward.shape[0]) for r in replays)
    plan = loop.schedule(True, rows_b, min(R * len(x) for x in stored), updates, s0.tqc.batch, s0.tqc.n_step, ws["fill"], ws["critic_steps"] * every, ws["actor_ramp"] * every)  # every-th environment step learns
    learn_from, actor_from, full_from = plan
    print(f"warm start: the replay fills to {100 * ws['fill']:.0f}% (step {learn_from:,}), then the critics learn alone for {ws['critic_steps']:,} gradient steps (to step {actor_from:,}), then the actors and temperatures ramp up to full over {ws['actor_ramp']:,} (to step {full_from:,})")
    weights = np.stack([s.weights for s in settings])
    h_from = [s.tqc.target_entropy for s in settings]
    chunk = make_chunk(cfgs, env_cfg, seats, weights, p["chunk"], updates, plan, stored, stride=every, entropy_from=h_from, entropy_ramp=entropy_ramp)
    if entropy_ramp and sb.get("entropy_shift"):
        print(f"target entropy: from {', '.join(f'{h:g}' for h in h_from)} to {', '.join(f'{c.target_entropy:g}' for c in cfgs)} over {entropy_ramp:,} steps from step {actor_from:,}")
    if sb.get("actor_lr"):
        print(f"actor learning rate {cfgs[0].actor_lr:g} (critics and temperature {cfgs[0].lr:g})")
    print(f"critics: {cfgs[0].n_step}-step targets ({cfgs[0].n_step * de.STEP_S:g} s of real rewards before their own guess)", flush=True)
    fast = [n for n, on in (("one encoder shared by the critics", cfgs[0].shared_encoder), ("bfloat16 dense layers", cfgs[0].encoder.bf16), (f"lidar {env_cfg.march_steps} steps per beam", env_cfg.march_steps != de.MARCH_STEPS)) if on]
    if fast:
        print(f"speed options: {', '.join(fast)}", flush=True)
    if cfgs[0].disagree > 0:
        print(f"disagreement penalty {cfgs[0].disagree:g}: the actor rates an action at the critics' mean less {cfgs[0].disagree:g} x their spread", flush=True)
    if env_cfg.start_tyres:
        print(f"training races start on random dry tyres: soft/medium/hard {'/'.join(f'{x:g}' for x in env_cfg.start_tyres)} (evals: mediums)", flush=True)
    group = np.where(seats < 0, len(names), seats)
    seats_per_group = R * np.bincount(group, minlength=len(names) + 1)

    outs = {s.profile: Path(a.out) / s.profile / sub for s in settings}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    log_file = open(Path(a.out) / f"{sub}-log.jsonl", "a")
    prov = checkpoint.provenance()
    log_file.write(json.dumps({"kind": "start", "argv": argv if argv is not None else sys.argv[1:], "reward_scale": s0.reward_scale, "profiles": list(names), "stage": label, "seats_per_group": [int(x) for x in seats_per_group], **prov}) + "\n")
    if prov.get("dirty"):
        print(f"note: the code has uncommitted changes (git {prov.get('git', '?')[:10]}): this run can't be reproduced from git alone")
    budget = int(sb.get("steps", 0)) or p["steps_b"]  # the stage's environment steps in all, over every resume
    # --resume carries the step count on (the replay starts empty again, so the warm start runs on the chunk's own count)
    start = min(checkpoint.meta(outs[s.profile] / "latest.npz").get("env_steps", 0) for s in settings) if a.resume else 0
    if start:
        print(f"resuming at step {start:,} of {budget:,}")
    eval_every = int(sb.get("eval_every", 0)) or p["eval_every_b"]
    snap_on = sb.get("snapshot_on", "best")
    if snap_on == "none":
        print(f"snapshot pool: fixed, the {n_frozen} frozen cars keep the drivers it started with")
    if snap_on == "best":
        print(f"snapshot pool: each new best eval's driver replaces the oldest snapshot of its own personality; {n_frozen // K or 1} frozen cars each")
    if snap_on != "none":
        for prof in sorted(set(owners)):
            k_own = owners.count(prof)
            print(f"  {prof}: {k_own} snapshot{'s' if k_own > 1 else ''}, " + ("replaced by its own bests" if prof in names else "fixed (not learning in this run)"))
        for nm in names:
            if nm not in owners:
                print(f"  {nm}: no snapshot of its own in the pool, so its bests don't go in")
    # the averaged actor: each profile's actor weights averaged with a half-life of `actor_average` steps, not the
    # actor at that moment, which may be at the bottom of a swing. The frozen cars' snapshots and the evals take
    # it; training and the checkpoints keep the current actor. 0: the current actor everywhere
    half = float(sb.get("actor_average", 0))
    avg_share = 1.0 - 0.5 ** (p["chunk"] / half) if half > 0 else None
    averaged = [jax.tree.map(jnp.copy, ag.actor) for ag in league.agents]  # copies: each chunk donates the league
    if avg_share and a.resume:  # a stopped run's averaged actors carry on (saved with its latest.npz, the same step)
        for i, s in enumerate(settings):
            kept, latest = outs[s.profile] / "latest-averaged.npz", outs[s.profile] / "latest.npz"
            if s.profile not in inits and kept.exists() and checkpoint.meta(kept).get("env_steps") == checkpoint.meta(latest).get("env_steps"):
                averaged[i] = checkpoint.load(kept, tqc.init(jax.random.PRNGKey(0), s.tqc))[0].actor
                print(f"{s.profile:>10}: averaged actor from {kept}")
    if avg_share:
        print(f"averaged actors ({half:,.0f}-step half-life): the frozen snapshots and the evals")
    else:
        print("no averaged actors: the evals, bests and frozen snapshots take the training actor itself")
    next_eval = (start // eval_every + 1) * eval_every  # in the stage's own steps (resumes included)
    rotate_every = p.get("rotate_every", 0) if a.rotate_every is None else a.rotate_every
    next_rotate = rotate_every if rot.rotates and rotate_every else None
    print(f"training tracks: {rot.describe()}")
    # an earlier run's best (same eval) stays the bar, so a new run's first eval can't overwrite a better checkpoint
    eval_seeds = int(sb.get("eval_seeds", p["eval_seeds"]))  # each eval track this many times, in parallel
    eval_meta = {"stage": label, "eval_kind": "weekend-orders" if st.orders else "weekend", "eval_laps": p["eval_laps"], "eval_version": evaluate.EVAL_VERSION}
    if eval_seeds != 1:  # a bigger eval is another eval: an old one-weekend best is no bar
        eval_meta["eval_seeds"] = eval_seeds
    eval_meta["eval_tracks"] = [t.name for t in eval_env.tracks]  # other tracks, another eval
    orders = {k: tuple(v) for k, v in sb.get("best_order", {}).items()}  # each profile's checks for a best (`evaluate.better_ordered`)
    for s_ in settings:
        if s_.profile in orders:
            print(f"{s_.profile:>10}: a new best by {' > '.join(orders[s_.profile])} > reward per step")
    # different race rules (`env.stage_race`) are a different eval: old bests without them are no bar
    eval_meta.update({f"eval_{k}": float(sb[k]) for k in ("stuck_s", "stuck_track_s") if k in sb})
    if sb.get("sc_band_kmh"):
        eval_meta["eval_sc_band_kmh"] = float(sb["sc_band_kmh"])
    # the eval's reference field: the stage's starting Middle, frozen (a copy beside the stage's log, kept on --resume),
    # so a best means better than where the stage started; --reference: another checkpoint, or "none"
    ref_src = sb.get("eval_reference") or inits.get("middle") or a.init_all or str(Path(a.out) / "middle" / st.before / "best.npz")
    reference, ref_file = None, Path(a.out) / f"{sub}-reference.npz"
    if ref_src != "none":
        if not (a.resume and ref_file.exists()):
            shutil.copyfile(ref_src, ref_file)
        ref_info = checkpoint.meta(ref_file)
        reference = checkpoint.load(ref_file, tqc.init(jax.random.PRNGKey(0), s0.tqc))[0].actor
        eval_meta["eval_reference"] = f"{ref_info.get('profile')} stage {ref_info.get('stage')} at {ref_info.get('env_steps', 0)}"
        print(f"eval reference field: {ref_file} (from {ref_src}: {eval_meta['eval_reference']} steps)")
    weights_of = [s.weights for s in settings]
    best: dict[str, dict | None] = {s.profile: checkpoint.best_bar(outs[s.profile] / "best.npz", **eval_meta) for s in settings}
    for name, bar in best.items():
        if bar:
            print(f"{name:>10}: best so far {evaluate.summary_mixed(bar)}")

    def run_eval(league, n: int):
        """The eval weekend at step n: logs it, saves latest.npz and, if it beats the bar, best.npz (the current actor)
        and bests/best-<n>.npz (the averaged actor the eval drove); with snapshot_on "best" (and n > 0) that driver
        replaces the oldest snapshot of its own personality. Returns the league."""
        drivers = averaged if avg_share else [ag.actor for ag in league.agents]
        res = season.eval_weekend(eval_env, drivers, cfgs[0], eval_seeds, log=print, names=names, weights=weights_of, reference=reference, march_steps=env_cfg.march_steps, orders=env_cfg.random_orders)
        if "reference" in res["profiles"]:
            print(f"  eval  reference: {evaluate.summary_mixed(res['profiles']['reference'])}", flush=True)
        log_file.write(json.dumps({"kind": "eval", "env_steps": n, **res}) + "\n")
        log_file.flush()
        for i, (s, ag) in enumerate(zip(settings, league.agents)):
            r = res["profiles"][s.profile]
            meta = {"profile": s.profile, "preset": a.preset, **eval_meta, "env_steps": n, "eval": r, "reward_scale": s.reward_scale, **prov}
            checkpoint.save(outs[s.profile] / "latest.npz", ag, meta)  # the current actor, to carry on from
            if avg_share:  # ... and the averaged one beside it, for --resume
                checkpoint.save(outs[s.profile] / "latest-averaged.npz", ag._replace(actor=drivers[i]), {**meta, "actor": "averaged"})
            improved = evaluate.better_ordered(r, best[s.profile], orders.get(s.profile, evaluate.DEFAULT_ORDER))  # the profile's own checks, in its order
            if improved:
                best[s.profile] = r
                checkpoint.save(outs[s.profile] / "best.npz", ag, meta)  # the current actor (the eval drove the averaged one)
                checkpoint.save(outs[s.profile] / "bests" / f"best-{n}.npz", ag._replace(actor=drivers[i]), {**meta, "actor": "averaged"})
                slot = oldest_slot(owners, ages, s.profile)
                if snap_on == "best" and n > start and slot is not None:  # the first eval's driver is the starting checkpoint
                    league = push_snapshot(league, slot, drivers[i])
                    ages[slot] = max(ages) + 1
                    print(f"  pool: the {n:,}-step best into slot {slot} (one of {owners.count(s.profile)} {s.profile} slots)", flush=True)
            print(f"  eval {s.profile:>10}: {evaluate.summary_mixed(r)}{'  (best)' if improved else ''}", flush=True)
        return league

    revive_on, revived = bool(sb.get("revive", False)), False

    def revive_dead(league, averaged, n: int):
        """Each learner's dead head units made new (`f1sim.revive`), in its actor and averaged actor alike and in its
        critics and their targets, judged on a sample of its replay. The drivers compute what they did."""
        agents, averaged = list(league.agents), list(averaged)
        for i, (s_, ag) in enumerate(zip(settings, agents)):
            b = rp.sample(league.replays[min(i, len(league.replays) - 1)], jax.random.fold_in(league.key, 7919 + n), 16384, cfgs[i].n_step, cfgs[i].gamma)
            if not bool(b.ok):
                continue
            obs, act = np.asarray(b.obs, np.float32), np.asarray(b.action, np.float32)
            actor, avg, ra = revive.revive(ag.actor, cfgs[i].encoder, obs, twin=averaged[i] if avg_share else None, seed=n + 2 * i)
            critic, target, rc = revive.revive(ag.critic, cfgs[i].encoder, obs, act, twin=ag.target, seed=n + 2 * i + 1)
            agents[i] = ag._replace(actor=actor, critic=critic, target=target)
            if avg_share:
                averaged[i] = avg
            print(f"  units {s_.profile:>9}: actor {revive.describe(ra)} | critics {revive.describe(rc)}", flush=True)
            log_file.write(json.dumps({"kind": "revive", "env_steps": n, "profile": s_.profile, "actor": ra, "critics": rc}) + "\n")
        return league._replace(agents=tuple(agents)), averaged

    if sb.get("eval_first") and any(bar is None for bar in best.values()):  # the starting drivers set the bar
        print("eval at step 0: the starting drivers set the bar", flush=True)
        league = run_eval(league, start)
    t_start = time.time()
    stop = StopFlag()
    while start + int(league.steps) < budget and not stop.asked:
        t0 = time.time()
        league, (stats, metrics) = chunk(league, bank)
        jax.block_until_ready(league.steps)
        here = int(league.steps)  # this run's steps (the warm start counts these)
        n = start + here  # the stage's steps
        rec = chunk_record(stats, metrics, names, time.time() - t0, p["chunk"], seats_per_group)
        log_file.write(json.dumps({"kind": "chunk", "env_steps": n, **rec}) + "\n")
        log_file.flush()
        print(f"{n:>9,} steps {rec['env_steps_per_s']:5.0f}/s | " + " | ".join(line_part(name, r) for name, r in rec["profiles"].items()), flush=True)
        if avg_share:
            averaged = [average_in(m, ag.actor, avg_share) for m, ag in zip(averaged, league.agents)]

        if revive_on and not revived and here >= learn_from:  # a run's start, once the replay has filled to the warm start's
            # level: judged on a sample of the first few seconds of racing, units that fire only later in a race (the safety car,
            # a pit stop) look dead and would lose what they learned (the first try, at 1,000 steps: 1,001 critic units)
            league, averaged = revive_dead(league, averaged, n)
            revived = True
        if next_rotate and n >= next_rotate:  # a fresh set of training tracks
            next_rotate += rotate_every
            bank, env_state, obs, hit = rotate_races(rot, league.env, league.obs, jax.random.fold_in(league.key, n), env_cfg)
            cut = tuple(rp.cut_last(buf, jnp.asarray(np.repeat(hit, len(stored[q])))) for q, buf in enumerate(league.replays))
            league = league._replace(env=env_state, obs=obs, replays=cut)
            trim_heap()
            print(f"  tracks: {rot.describe()}; {int(hit.sum())} races restarted", flush=True)
        if n >= next_eval:
            next_eval += eval_every
            if here > plan.actor_from:  # before it the actors haven't moved: the same drivers, the same eval
                if revive_on:  # before every eval: the units that died since, and the count for the log
                    league, averaged = revive_dead(league, averaged, n)
                league = run_eval(league, n)
                trim_heap()
    for i, (s, ag) in enumerate(zip(settings, league.agents)):
        meta = {"profile": s.profile, "preset": a.preset, "stage": label, "env_steps": start + int(league.steps), "reward_scale": s.reward_scale, **prov}
        checkpoint.save(outs[s.profile] / "latest.npz", ag, meta)
        if avg_share:
            checkpoint.save(outs[s.profile] / "latest-averaged.npz", ag._replace(actor=averaged[i]), {**meta, "actor": "averaged"})
    print(f"stage {label}: {int(league.steps):,} steps in {(time.time() - t_start) / 60:.1f} min (the stage at {start + int(league.steps):,} of {budget:,})")
    log_file.close()


if __name__ == "__main__":
    main()
