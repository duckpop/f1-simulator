"""Stage B training (Phase 2, section 3.2): the full grid, the four profiles learning together.

    f1sim-train-b --tracks tracks --preset cluster                  # each profile from its stage A best
    f1sim-train-b --generate 4 --eval-tracks 2 --init-all runs/middle/stage-a/best.npz --steps 2000   # smoke run

- 22 cars per race on the 12-15 m tracks, with every rule f1env has: overtaking, dirty air, slipstream, DRS, contact,
  the safety car, blue flags. Seats are fixed by car number (the grid is random every race): `per_profile` cars of each
  profile learn (4 x 4 = 16), and the other 6 (27%) drive frozen snapshots of earlier checkpoints, so the four don't
  only learn to beat each other's current habits.
- Each profile has its own networks, optimisers and replay buffer. One batch of races trains all four: each car's steps
  go to its profile's buffer, with its profile's reward weights, and every profile takes its gradient steps each
  environment step.
- The snapshot pool starts with the four stage A bests; every `snapshot_every` steps it takes a copy of one profile's
  current actor (in turn), replacing the oldest. A frozen car drives one snapshot (its mean action) for a whole
  session, drawn again when the session restarts.
- 20% of sessions are short qualifying sessions (`quali_share`), so flying laps and the "session" input are learned.
- The eval: mixed races on the eval tracks, every seat a learning profile (5-6 cars each), the default car, no
  respawn. Each profile keeps its best checkpoint (most points per race, then finishes) in runs/<profile>/stage-b/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from f1env.sensors import make_bank

from .. import checkpoint, loop, nets, tqc
from .. import replay as rp
from . import env as de
from . import evaluate, rewards
from .config import PROFILES, load
from .tracks import eval_tracks, training_tracks

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


def make_chunk(cfgs, env_cfg: de.EnvConfig, seats: np.ndarray, weights: np.ndarray, steps: int, updates: int, learn_from: int):
    """cfgs: a TQCConfig per profile; weights (profiles, terms). One jitted call runs `steps` environment steps, each
    followed by `updates` gradient steps for every profile."""
    P, (R, C) = len(cfgs), (env_cfg.races, env_cfg.cars)
    idx = [np.flatnonzero(seats == p) for p in range(P)]
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
        on = c.steps >= learn_from
        for p in range(P):
            sl = idx[p]
            flat = lambda x: x[:, sl].reshape((-1,) + x.shape[2:])
            buf = rp.add(c.replays[p], flat(c.obs), flat(actions), flat(reward), flat(out.valid), flat(out.end))
            agent, m = loop.maybe_learn(on, c.agents[p], cfgs[p], buf, keys[3 + P + p], updates)
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
            "reward": jnp.einsum("rc,cg->g", reward, onehot),
            "races": info["races"],
        }
        metrics = {k: jnp.stack([m[k] for m in ms]) for k in tqc.METRICS}
        metrics["learning"] = on.astype(jnp.float32)
        return League(env, nxt, tuple(agents), tuple(replays), c.pool, pool_of, key, c.steps + 1), (stats, metrics)

    @partial(jax.jit, donate_argnums=0)
    def chunk(league: League, bank):
        return jax.lax.scan(partial(body, bank), league, None, length=steps)

    return chunk


def push_snapshot(league: League, slot: int, actor) -> League:
    """Copy an actor into pool slot `slot`."""
    pool = jax.tree.map(lambda s, a: s.at[slot].set(a), league.pool, actor)
    return league._replace(pool=pool)


def chunk_record(stats, metrics, profiles, elapsed: float, steps: int) -> dict:
    amounts = np.asarray(stats["amounts"]).sum(axis=0)  # (groups, terms)
    valid = np.asarray(stats["valid"]).sum(axis=0)
    laps = np.asarray(stats["laps"]).sum(axis=0)
    reward = np.asarray(stats["reward"]).sum(axis=0)
    learning = np.asarray(metrics["learning"]) > 0
    t = {name: i for i, name in enumerate(rewards.TERMS)}
    rec = {"env_steps_per_s": steps / elapsed, "races": float(np.asarray(stats["races"]).sum()), "profiles": {}}
    for g, name in enumerate([*profiles, "frozen"]):
        v = max(valid[g], 1.0)
        r = {
            "transitions": float(valid[g]),
            "laps": float(laps[g]),
            "crashes": float(amounts[g, t["crash"]]),
            "crashes_per_lap": float(amounts[g, t["crash"]] / max(laps[g], 1.0)),
            "overtakes": float(amounts[g, t["overtake"]]),
            "places_lost": float(amounts[g, t["place_lost"]]),
            "holding_s": float(amounts[g, t["holding"]]),
            "finishes": float(amounts[g, t["finish"]]),
        }
        if g < len(profiles):
            r["reward_per_step"] = float(reward[g] / v)
            if learning.any():
                r.update({k: float(np.asarray(metrics[k])[learning, g].mean()) for k in tqc.METRICS})
        rec["profiles"][name] = r
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage B: the four profiles learning together on full grids.")
    ap.add_argument("--preset", default="laptop")
    ap.add_argument("--tracks", help="track pool directory (f1env-tracks --out ...)")
    ap.add_argument("--generate", type=int, default=0, help="no pool: generate this many training tracks")
    ap.add_argument("--eval-tracks", type=int, default=10)
    ap.add_argument("--init-all", help="start every profile from this checkpoint (default: each one's stage A best)")
    ap.add_argument("--steps", type=int, help="override steps_b (environment steps)")
    ap.add_argument("--eval-every", type=int, help="override eval_every (environment steps)")
    ap.add_argument("--snapshot-every", type=int, help="override snapshot_every (environment steps)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.tracks is None and not a.generate:
        ap.error("give a track pool (--tracks) or --generate N")

    settings = [load(p, a.preset) for p in PROFILES]
    s0 = settings[0]
    p, sb = s0.preset, s0.stage_b
    cfgs = [s.tqc for s in settings]
    seats = seat_layout(CARS, sb["per_profile"])
    env_cfg = s0.env._replace(races=p["races_b"], cars=CARS, quali_share=sb["quali_share"], quali_s=sb["quali_s"])
    R = env_cfg.races

    key = jax.random.PRNGKey(a.seed)
    agents, replays = [], []
    for s in settings:
        template = tqc.init(jax.random.PRNGKey(0), s.tqc)
        init = a.init_all or str(Path(a.out) / s.profile / "stage-a" / "best.npz")
        base, info = checkpoint.load(init, template)
        agents.append(tqc.from_networks(s.tqc, base.actor, base.critic, base.target, base.log_alpha))
        slots = R * int((seats == PROFILES.index(s.profile)).sum())
        replays.append(rp.new(rp.rows_for(p.get("replay_b", p["replay"]), slots), slots, de.OBS, de.ACTIONS))
        print(f"{s.profile:>10}: from {init} ({info.get('profile')}, {info.get('env_steps', 0):,} steps)  d={s.tqc.drop} H0={s.tqc.target_entropy} gamma={s.tqc.gamma}")
    K = sb["snapshots"]
    pool = stack([agents[k % len(agents)].actor for k in range(K)])
    n_frozen = int((seats < 0).sum())
    print(f"{R} races x {CARS} cars: {sb['per_profile']} per profile learning, {n_frozen} frozen ({K} snapshots); {100 * sb['quali_share']:.0f}% qualifying")

    tracks = training_tracks(a.tracks, "normal", a.generate or p["tracks"], seed=a.seed)
    bank, eval_bank = make_bank(tracks), make_bank(eval_tracks(a.tracks, "normal", a.eval_tracks))
    key, k_reset, k_pool = jax.random.split(key, 3)
    env_state, obs = de.compiled("reset", env_cfg)(bank, k_reset)
    league = League(env_state, obs, tuple(agents), tuple(replays), pool, jax.random.randint(k_pool, (R, n_frozen), 0, K), key, jnp.zeros((), jnp.int32))
    learn_from = max(int(sb["warmup"]) // (R * sb["per_profile"]), s0.tqc.batch // (R * sb["per_profile"]) + s0.tqc.n_step + 1)
    weights = np.stack([s.weights for s in settings])
    chunk = make_chunk(cfgs, env_cfg, seats, weights, p["chunk"], p["updates"], learn_from)

    outs = {s.profile: Path(a.out) / s.profile / "stage-b" for s in settings}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    log_file = open(Path(a.out) / "stage-b-log.jsonl", "a")
    budget = a.steps or p["steps_b"]
    eval_every, snap_every = a.eval_every or p["eval_every"], a.snapshot_every or sb["snapshot_every"]
    next_eval, next_snap, slot, turn = eval_every, snap_every, 0, 0
    best: dict[str, dict | None] = {s.profile: None for s in settings}
    t_start = time.time()
    while int(league.steps) < budget:
        t0 = time.time()
        league, (stats, metrics) = chunk(league, bank)
        jax.block_until_ready(league.steps)
        n = int(league.steps)
        rec = chunk_record(stats, metrics, PROFILES, time.time() - t0, p["chunk"])
        log_file.write(json.dumps({"kind": "chunk", "env_steps": n, **rec}) + "\n")
        log_file.flush()
        parts = []
        for name, r in rec["profiles"].items():
            parts.append(f"{name[:3]} cr {r['crashes']:3.0f} lap {r['laps']:4.0f} ov {r['overtakes']:3.0f}" + (f" r {r['reward_per_step']:5.2f}" if "reward_per_step" in r else ""))
        print(f"{n:>9,} steps {rec['env_steps_per_s']:5.0f}/s | " + " | ".join(parts), flush=True)

        if n >= next_snap:  # a copy of one profile's actor into the pool, replacing the oldest
            next_snap += snap_every
            league = push_snapshot(league, slot, league.agents[turn].actor)
            slot, turn = (slot + 1) % K, (turn + 1) % len(PROFILES)
        if n >= next_eval:
            next_eval += eval_every
            res = evaluate.mixed(eval_bank, [ag.actor for ag in league.agents], cfgs[0], p["eval_laps"], p["eval_seeds"])
            log_file.write(json.dumps({"kind": "eval", "env_steps": n, **res}) + "\n")
            log_file.flush()
            for s, ag in zip(settings, league.agents):
                r = res["profiles"][s.profile]
                meta = {"profile": s.profile, "preset": a.preset, "stage": "B", "env_steps": n, "eval": r}
                checkpoint.save(outs[s.profile] / "latest.npz", ag, meta)
                improved = evaluate.better_mixed(r, best[s.profile])
                if improved:
                    best[s.profile] = r
                    checkpoint.save(outs[s.profile] / "best.npz", ag, meta)
                print(f"  eval {s.profile:>10}: {evaluate.summary_mixed(r)}{'  (best)' if improved else ''}", flush=True)
    for s, ag in zip(settings, league.agents):
        checkpoint.save(outs[s.profile] / "latest.npz", ag, {"profile": s.profile, "preset": a.preset, "stage": "B", "env_steps": int(league.steps)})
    print(f"stage B: {int(league.steps):,} steps in {(time.time() - t_start) / 60:.1f} min")
    log_file.close()


if __name__ == "__main__":
    main()
