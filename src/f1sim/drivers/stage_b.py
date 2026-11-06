"""Stage B training (Phase 2, section 3.2): the full grid, the four profiles learning together.

    f1sim-train-b --tracks tracks --preset cluster                  # each profile from its stage A best
    f1sim-train-b --generate 4 --eval-tracks 2 --init-all runs/middle/stage-a/best.npz --steps 2000   # smoke run
    f1sim-train-b --tracks tracks --init smooth=runs/middle/stage-b/best.npz ...   # a profile from a given checkpoint
    f1sim-train-b --tracks tracks --profiles middle --out runs/together --init middle=runs/middle/stage-a/best.npz
                                                                    # one profile alone on the grid: 16 learning, 6 frozen

- 22 cars per race on the 12-15 m tracks, with every rule f1env has: overtaking, dirty air, slipstream, DRS, contact,
  the safety car, blue flags. Seats are fixed by car number (the grid is random every race): `per_profile` cars of each
  profile learn (4 x 4 = 16), and the other 6 (27%) drive frozen snapshots of earlier checkpoints, so the four don't
  only learn to beat each other's current habits.
- `--profiles` picks the profiles that learn (default all four); the 16 learning seats are split between them
  (`--per-profile` overrides), so `--profiles middle` races 16 learning Middles against 6 frozen earlier ones.
- Each profile's target entropy is `entropy_shift` (-1) lower than in stage A: less action noise, so training drives
  closer to the noise-free eval (at stage A's noise tyres lasted half as long).
- Each profile has its own networks, optimisers and replay buffer. One batch of races trains all four: each car's steps
  go to its profile's buffer, with its profile's reward weights, and every profile takes its gradient steps each
  environment step.
- The snapshot pool starts with the four stage A bests; every `snapshot_every` steps it takes a copy of one profile's
  current actor (in turn), replacing the oldest. A frozen car drives one snapshot (its mean action) for a whole
  session, drawn again when the session restarts.
- 20% of sessions are short qualifying sessions (`quali_share`), so flying laps and the "session" input are learned.
- Field sizes and spread starts (since 2026-11-06): each session has 6, 12, 18 or 22 cars (`field_sizes`), and half
  the races (`spread_share`) start with the cars spread around the lap at speed instead of on the grid. Every race was
  a 22-car standing start before: the learners lasted ~1.8 laps, mostly in lap-1 pile-ups and the safety car periods
  they cause, and lost driving skill (together6's lone wall crashes ~6x the reference's). v2 trained on 3-car races.
- The eval: a full weekend on every eval track (Q1, Q2 and Q3 set the grid, then a race of `eval_laps` laps), the
  default car, no respawn: the even cars the learning profiles in turn, the odd ones (one in every team) a fixed
  reference driver (`eval_reference`, or `--reference`), so points mean the same from eval to eval and a profile can't
  look better because another got worse. Then each profile alone on the eval tracks for `solo_laps` laps: its pace,
  tyre life and crashes apart from the field. Each profile keeps its best checkpoint (`evaluate.better_mixed`: points
  against the reference, then the distance completed, then its own reward per step) in runs/<profile>/stage-b/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")  # JAX stops at 75% of the GPU by default

import jax
import jax.numpy as jnp
import numpy as np

from .. import checkpoint, loop, nets, tqc
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


def init_path(profile: str, out, inits: dict, resume: bool = False, init_all: str | None = None) -> str:
    """The checkpoint a profile starts from: its own --init, else its stage B latest.npz (--resume), --init-all, or
    its stage A best."""
    if profile in inits:
        return inits[profile]
    if resume:
        return str(Path(out) / profile / "stage-b" / "latest.npz")
    return init_all or str(Path(out) / profile / "stage-a" / "best.npz")


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


def make_chunk(cfgs, env_cfg: de.EnvConfig, seats: np.ndarray, weights: np.ndarray, steps: int, updates: int, learn_from: int, stored=None, actor_from: int = 0):
    """cfgs: a TQCConfig per profile; weights (profiles, terms). One jitted call runs `steps` environment steps, each
    followed by `updates` gradient steps for every profile. stored: per profile, the car numbers whose steps go to its
    replay (default: all of its seats); actor_from: environment steps before the actors and temperatures learn (until
    then the critics learn alone, from learn_from)."""
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
            a = tqc.act(c.agents[p].actor, cfgs[p], obs[:, idx[p]].reshape(-1, de.OBS), keys[3 + p])
            actions = actions.at[:, idx[p]].set(a.reshape(R, len(idx[p]), de.ACTIONS))
        if frozen.size:
            o = obs[:, frozen]
            every = jax.vmap(lambda prm: nets.actor_mean(prm, enc, o))(c.pool)  # (K, R, F, actions)
            a = every[c.pool_of, jnp.arange(R)[:, None], jnp.arange(frozen.size)[None]]
            actions = actions.at[:, frozen].set(a)

        env, nxt, out, info = de.step(bank, c.env, actions, k_env, env_cfg)
        reward = jnp.sum(out.amounts * w_car[None], axis=-1)
        agents, replays, ms = [], [], []
        on, actor_on = c.steps >= learn_from, c.steps >= actor_from
        for p in range(P):
            sl = stored[p]
            flat = lambda x: x[:, sl].reshape((-1,) + x.shape[2:])
            buf = rp.add(c.replays[p], flat(c.obs), flat(actions), flat(reward), flat(out.valid), flat(out.end))
            agent, m = loop.maybe_learn(on, c.agents[p], cfgs[p], buf, keys[3 + P + p], updates, actor_on)
            agents.append(agent)
            replays.append(buf)
            ms.append(m)
        restarted = env.tick == 0
        pool_size = jax.tree.leaves(c.pool)[0].shape[0]
        pool_of = jnp.where(restarted[:, None], jax.random.randint(k_pool, c.pool_of.shape, 0, pool_size), c.pool_of)
        stats = {
            "amounts": jnp.einsum("rck,cg->gk", out.amounts, onehot),  # (groups, terms)
            "valid": jnp.einsum("rc,cg->g", out.valid.astype(jnp.float32), onehot),
            "laps": jnp.einsum("rc,cg->g", out.lap_done.astype(jnp.float32), onehot),
            "crashes": jnp.einsum("rc,cg->g", out.crashed.astype(jnp.float32), onehot),
            "hit": jnp.einsum("rc,cg->g", out.hit.astype(jnp.float32), onehot),
            "reward": jnp.einsum("rc,cg->g", reward, onehot),
            "races": info["races"],
        }
        metrics = {k: jnp.stack([m[k] for m in ms]) for k in tqc.METRICS}
        metrics["learning"] = on.astype(jnp.float32)
        metrics["actor"] = (on & actor_on).astype(jnp.float32)
        return League(env, nxt, tuple(agents), tuple(replays), c.pool, pool_of, key, c.steps + 1), (stats, metrics)

    @partial(jax.jit, donate_argnums=0)
    def chunk(league: League, bank):
        return jax.lax.scan(partial(body, bank), league, None, length=steps)

    return chunk


def push_snapshot(league: League, slot: int, actor) -> League:
    """Copy an actor into pool slot `slot`."""
    pool = jax.tree.map(lambda s, a: s.at[slot].set(a), league.pool, actor)
    return league._replace(pool=pool)


def chunk_record(stats, metrics, profiles, elapsed: float, steps: int, seats_per_group=None) -> dict:
    """seats_per_group: (profiles + 1,) cars in each group over all races (the frozen ones last), for the share alive."""
    amounts = np.asarray(stats["amounts"]).sum(axis=0)  # (groups, terms)
    valid = np.asarray(stats["valid"]).sum(axis=0)
    laps = np.asarray(stats["laps"]).sum(axis=0)
    crashes = np.asarray(stats["crashes"]).sum(axis=0)
    hit = np.asarray(stats["hit"]).sum(axis=0)
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
        if seats_per_group is not None:
            r["alive"] = float(valid[g] / max(seats_per_group[g] * steps, 1))
        if g < len(profiles):
            r["reward_per_step"] = float(reward[g] / v)
            if learning.any():
                r.update({k: float(np.asarray(metrics[k])[learning, g].mean()) for k in tqc.METRICS})
                r["critics_only"] = bool(not (np.asarray(metrics["actor"])[learning] > 0).any())
        rec["profiles"][name] = r
    return rec


def line_part(name: str, r: dict) -> str:
    """One group's part of the log line: the share of its seats alive, crashes (per lap; how many it was hit in), black
    flags, laps, overtakes; for a learning profile its reward per step and the critics' Q, alpha and entropy."""
    out = f"{name[:3]} alive {100 * r.get('alive', float('nan')):3.0f}% cr {r['crashes']:3.0f} ({r['crashes_per_lap']:.2f}/lap, hit {r.get('hit', 0):.0f}) bf {r.get('black_flags', 0):.0f} lap {r['laps']:4.0f} ov {r['overtakes']:3.0f}"
    if "reward_per_step" in r:
        out += f" r {r['reward_per_step']:6.3f}"
    if "q" in r:
        out += f" q {r['q']:6.1f} α {r['alpha']:.4f} H {r['entropy']:5.2f}" + (" (critics only)" if r.get("critics_only") else "")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage B: the four profiles learning together on full grids.")
    ap.add_argument("--preset", default="laptop")
    ap.add_argument("--tracks", help="track pool directory (f1env-tracks --out ...)")
    ap.add_argument("--generate", type=int, default=0, help="no pool: generate this many training tracks")
    ap.add_argument("--eval-tracks", type=int, default=10)
    ap.add_argument("--init-all", help="start every profile from this checkpoint (default: each one's stage A best)")
    ap.add_argument("--resume", action="store_true", help="continue every profile from its own stage B latest.npz (after a stop)")
    ap.add_argument("--init", action="append", default=[], metavar="PROFILE=PATH", help="start this profile from this checkpoint (repeatable; over --init-all and --resume)")
    ap.add_argument("--steps", type=int, help="override steps_b (environment steps)")
    ap.add_argument("--eval-every", type=int, help="override eval_every_b (environment steps)")
    ap.add_argument("--snapshot-every", type=int, help="override snapshot_every (environment steps)")
    ap.add_argument("--rotate-every", type=int, help="override rotate_every (environment steps; 0: keep one set of tracks)")
    ap.add_argument("--tracks-per-set", type=int, help="override tracks (two sets are on the GPU when rotating)")
    ap.add_argument("--replay", type=int, help="override replay_b (transitions per profile)")
    ap.add_argument("--profiles", default=",".join(PROFILES), help="the profiles that learn, comma-separated (default: all four)")
    ap.add_argument("--per-profile", type=int, help="learning cars per profile (default: 16 learning seats split between the profiles)")
    ap.add_argument("--reference", help='the eval\'s fixed reference driver (default: eval_reference in the config; "none": no reference field)')
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.tracks is None and not a.generate:
        ap.error("give a track pool (--tracks) or --generate N")
    names = tuple(a.profiles.split(","))
    if set(names) - set(PROFILES) or len(set(names)) != len(names):
        ap.error(f"--profiles takes some of {', '.join(PROFILES)}, each once: {a.profiles}")
    inits = dict(x.split("=", 1) for x in a.init if "=" in x)
    if len(inits) != len(a.init) or set(inits) - set(names):
        ap.error(f"--init takes PROFILE=PATH, once per profile ({', '.join(names)}): {' '.join(a.init)}")

    settings = [load(p, a.preset) for p in names]
    s0 = settings[0]
    p, sb = s0.preset, s0.stage_b
    p.update({k: v for k, v in (("tracks", a.tracks_per_set), ("replay_b", a.replay)) if v})
    cfgs = [s.tqc._replace(target_entropy=s.tqc.target_entropy + sb.get("entropy_shift", 0.0)) for s in settings]  # less noise
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
    )
    R = env_cfg.races

    key = jax.random.PRNGKey(a.seed)
    agents, replays, done = [], [], 0
    for i, s in enumerate(settings):
        template = tqc.init(jax.random.PRNGKey(0), s.tqc)
        init = init_path(s.profile, a.out, inits, a.resume, a.init_all)
        base, info = checkpoint.load(init, template, s.tqc)
        done += info.get("env_steps", 0)
        # the same networks: their optimisers carry on; deepened or widened ones start fresh ones
        exact = "deepened" not in info
        agent = base if exact else tqc.from_networks(cfgs[i], base.actor, base.critic, base.target, base.log_alpha)
        factor = s.reward_scale / float(info.get("reward_scale", 1.0))  # checkpoints before the scale: 1
        if factor != 1.0:
            agent = tqc.rescale(agent, factor)
        agents.append(agent)
        slots = R * len(stored[i])
        rows = rp.rows_for(p.get("replay_b", p["replay"]), slots)
        replays.append(rp.new(rows, slots, de.OBS, de.ACTIONS))
        notes = ["optimisers kept" if exact else "fresh optimisers"] + ([f"rewards x{factor:g} since: critics rescaled"] if factor != 1.0 else [])
        print(f"{s.profile:>10}: from {init} ({info.get('profile')}, {info.get('env_steps', 0):,} steps)  d={s.tqc.drop} H0={cfgs[i].target_entropy} gamma={s.tqc.gamma}  ({', '.join(notes)})")
        print(f"{'':>10}  replay: {len(stored[i])} of {int((seats == i).sum())} seats per race stored, {rows:,} steps ({rows * 0.05 / 60:.0f} min of racing)")
    K = sb["snapshots"]
    pool = stack([agents[k % len(agents)].actor for k in range(K)])
    n_frozen = int((seats < 0).sum())
    print(f"{R} races x {CARS} cars: {per} per profile learning, {n_frozen} frozen ({K} snapshots); {100 * sb['quali_share']:.0f}% qualifying")
    if env_cfg.field_sizes or env_cfg.spread_share:
        print(f"fields of {', '.join(map(str, env_cfg.field_sizes or (CARS,)))} cars; {100 * env_cfg.spread_share:.0f}% of races start spread around the lap at {env_cfg.spread_speed[0]:g}-{env_cfg.spread_speed[1]:g} x the reference pace")

    # the starting checkpoints' steps in the seed: a run continued from another takes the pool in a new order, not
    # the same first sets again
    rot = Rotation(a.tracks, "normal", a.generate or p["tracks"], seed=a.seed + done, generate=bool(a.generate))
    bank = rot.bank
    eval_env = season.weekend_env(eval_tracks(a.tracks, "normal", a.eval_tracks), p["eval_laps"])  # Q1-Q3, then the race
    key, k_reset, k_pool = jax.random.split(key, 3)
    env_state, obs = de.compiled("reset", env_cfg)(bank, k_reset, pick=rot.pick)
    league = League(env_state, obs, tuple(agents), tuple(replays), pool, jax.random.randint(k_pool, (R, n_frozen), 0, K), key, jnp.zeros((), jnp.int32))
    learn_from = max(int(sb.get("warmup_steps", 0)), s0.tqc.batch // min(R * len(x) for x in stored) + s0.tqc.n_step + 1)
    actor_from = learn_from + -(-int(sb.get("critic_warmup", 0)) // p["updates"])
    print(f"learning from step {learn_from:,}: the critics alone until step {actor_from:,}, then the actors and temperatures too")
    weights = np.stack([s.weights for s in settings])
    chunk = make_chunk(cfgs, env_cfg, seats, weights, p["chunk"], p["updates"], learn_from, stored, actor_from)
    group = np.where(seats < 0, len(names), seats)
    seats_per_group = R * np.bincount(group, minlength=len(names) + 1)

    outs = {s.profile: Path(a.out) / s.profile / "stage-b" for s in settings}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    log_file = open(Path(a.out) / "stage-b-log.jsonl", "a")
    prov = checkpoint.provenance()
    log_file.write(json.dumps({"kind": "start", "argv": argv if argv is not None else sys.argv[1:], "reward_scale": s0.reward_scale, "profiles": list(names), "seats_per_group": [int(x) for x in seats_per_group], **prov}) + "\n")
    if prov.get("dirty"):
        print(f"note: the code has uncommitted changes (git {prov.get('git', '?')[:10]}): this run can't be reproduced from git alone")
    budget = a.steps or p["steps_b"]
    eval_every, snap_every = a.eval_every or p.get("eval_every_b", p["eval_every"]), a.snapshot_every or sb["snapshot_every"]
    next_eval, next_snap, slot, turn = eval_every, snap_every, 0, 0
    rotate_every = p.get("rotate_every", 0) if a.rotate_every is None else a.rotate_every
    next_rotate = rotate_every if rot.rotates and rotate_every else None
    print(f"training tracks: {rot.describe()}")
    # an earlier run's best (same eval) stays the bar, so a new run's first eval can't overwrite a better checkpoint
    eval_meta = {"stage": "B", "eval_kind": "weekend", "eval_laps": p["eval_laps"], "eval_version": evaluate.EVAL_VERSION}
    ref_path = a.reference or sb.get("eval_reference")
    reference = None
    if ref_path and ref_path != "none":
        if Path(ref_path).exists():
            reference = checkpoint.load(ref_path, tqc.init(jax.random.PRNGKey(0), s0.tqc), s0.tqc)[0].actor
            print(f"eval reference field: {ref_path} ({checkpoint.meta(ref_path).get('env_steps', 0):,} steps)")
        else:
            print(f"eval reference {ref_path} not found: the eval races the profiles only")
    eval_meta["eval_reference"] = ref_path if reference is not None else None
    weights_of = [s.weights for s in settings]
    best: dict[str, dict | None] = {s.profile: checkpoint.best_bar(outs[s.profile] / "best.npz", **eval_meta) for s in settings}
    for name, bar in best.items():
        if bar:
            print(f"{name:>10}: best so far {evaluate.summary_mixed(bar)}")
    t_start = time.time()
    stop = StopFlag()
    while int(league.steps) < budget and not stop.asked:
        t0 = time.time()
        league, (stats, metrics) = chunk(league, bank)
        jax.block_until_ready(league.steps)
        n = int(league.steps)
        rec = chunk_record(stats, metrics, names, time.time() - t0, p["chunk"], seats_per_group)
        log_file.write(json.dumps({"kind": "chunk", "env_steps": n, **rec}) + "\n")
        log_file.flush()
        print(f"{n:>9,} steps {rec['env_steps_per_s']:5.0f}/s | " + " | ".join(line_part(name, r) for name, r in rec["profiles"].items()), flush=True)

        if next_rotate and n >= next_rotate:  # a fresh set of training tracks
            next_rotate += rotate_every
            bank, env_state, obs, hit = rotate_races(rot, league.env, league.obs, jax.random.fold_in(league.key, n), env_cfg)
            cut = tuple(rp.cut_last(buf, jnp.asarray(np.repeat(hit, len(stored[q])))) for q, buf in enumerate(league.replays))
            league = league._replace(env=env_state, obs=obs, replays=cut)
            print(f"  tracks: {rot.describe()}; {int(hit.sum())} races restarted", flush=True)
        if n >= next_snap:  # a copy of one profile's actor into the pool, replacing the oldest
            next_snap += snap_every
            league = push_snapshot(league, slot, league.agents[turn].actor)
            slot, turn = (slot + 1) % K, (turn + 1) % len(names)
        if n >= next_eval:
            next_eval += eval_every
            res = season.eval_weekend(eval_env, [ag.actor for ag in league.agents], cfgs[0], p["eval_seeds"], log=print, names=names, weights=weights_of, reference=reference)
            res["solo"] = {s.profile: evaluate.alone(eval_env.bank, ag.actor, cfgs[i], sb.get("solo_laps", 5), weights=s.weights) for i, (s, ag) in enumerate(zip(settings, league.agents))}
            if "reference" in res["profiles"]:
                print(f"  eval  reference: {evaluate.summary_mixed(res['profiles']['reference'])}", flush=True)
            log_file.write(json.dumps({"kind": "eval", "env_steps": n, **res}) + "\n")
            log_file.flush()
            for s, ag in zip(settings, league.agents):
                r = res["profiles"][s.profile]
                r = {**r, "solo": res["solo"][s.profile]}
                meta = {"profile": s.profile, "preset": a.preset, **eval_meta, "env_steps": n, "eval": r, "reward_scale": s.reward_scale, **prov}
                checkpoint.save(outs[s.profile] / "latest.npz", ag, meta)
                improved = evaluate.better_mixed(r, best[s.profile])
                if improved:
                    best[s.profile] = r
                    checkpoint.save(outs[s.profile] / "best.npz", ag, meta)
                print(f"  eval {s.profile:>10}: {evaluate.summary_mixed(r)}{'  (best)' if improved else ''}", flush=True)
                print(f"  solo {s.profile:>10}: {evaluate.summary(r['solo'])}", flush=True)
    for s, ag in zip(settings, league.agents):
        checkpoint.save(outs[s.profile] / "latest.npz", ag, {"profile": s.profile, "preset": a.preset, "stage": "B", "env_steps": int(league.steps), "reward_scale": s.reward_scale, **prov})
    print(f"stage B: {int(league.steps):,} steps in {(time.time() - t_start) / 60:.1f} min")
    log_file.close()


if __name__ == "__main__":
    main()
