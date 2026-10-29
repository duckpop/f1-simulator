"""TQC's check on a small standard control task before it meets F1, so a bug isn't mistaken for an F1 problem.

The classic pendulum swing-up (as in Gym's Pendulum-v1: 200-step episodes, torque in [-2, 2], cost on the angle, the
spin and the torque), written in JAX so it runs through the same collect-and-learn loop as the drivers. A random
policy scores about -1,200 per episode; a working TQC gets above -250 within ~25k transitions.

    f1sim-check-tqc            # trains for a few minutes and reports the average episode return
"""

from __future__ import annotations

import argparse
import os
import time
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from . import loop
from . import replay as rp
from . import tqc

EPISODE = 200
MAX_SPEED, MAX_TORQUE, DT, G = 8.0, 2.0, 0.05, 10.0


class Pendulum(NamedTuple):
    th: jax.Array  # (S,)
    thdot: jax.Array
    t: jax.Array
    ret: jax.Array  # this episode's return so far


def reset(key, n: int) -> Pendulum:
    k1, k2 = jax.random.split(key)
    z = jnp.zeros(n)
    return Pendulum(jax.random.uniform(k1, (n,), minval=-jnp.pi, maxval=jnp.pi), jax.random.uniform(k2, (n,), minval=-1.0, maxval=1.0), z.astype(jnp.int32), z)


def observe(p: Pendulum):
    return jnp.stack([jnp.cos(p.th), jnp.sin(p.th), p.thdot], axis=-1)


def step(p: Pendulum, action, key):
    u = MAX_TORQUE * action[:, 0]
    angle = jnp.mod(p.th + jnp.pi, 2 * jnp.pi) - jnp.pi
    reward = -(angle**2 + 0.1 * p.thdot**2 + 0.001 * u**2)
    thdot = jnp.clip(p.thdot + (3 * G / 2 * jnp.sin(p.th) + 3.0 * u) * DT, -MAX_SPEED, MAX_SPEED)
    moved = Pendulum(p.th + thdot * DT, thdot, p.t + 1, p.ret + reward)
    done = moved.t >= EPISODE
    fresh = reset(key, p.th.shape[0])
    nxt = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, moved)
    st = loop.Step(reward, jnp.ones_like(done), jnp.where(done, rp.CUT, rp.CONTINUE).astype(jnp.int8))
    info = {"episodes": jnp.sum(done).astype(jnp.float32), "return": jnp.sum(jnp.where(done, moved.ret, 0.0))}
    return nxt, observe(nxt), st, info


def check(steps: int = 6000, slots: int = 4, updates: int = 4, seed: int = 0, log=print) -> float:
    """Train TQC on the pendulum; the average return of the episodes in the last fifth of training."""
    cfg = tqc.TQCConfig(obs_dim=3, act_dim=1, gamma=0.99, n_step=1, target_entropy=-1.0, critic_hidden=(256, 256), batch=256)
    key = jax.random.PRNGKey(seed)
    k_env, k_agent, k_run = jax.random.split(key, 3)
    env = reset(k_env, slots)
    carry = loop.Carry(env, observe(env), tqc.init(k_agent, cfg), rp.new(steps + 8, slots, 3, 1), k_run, jnp.zeros((), jnp.int32))
    chunk_steps = 250
    chunk = loop.make_chunk(lambda s, a, k: step(s, a, k), cfg, chunk_steps, updates, warmup=250, learn_from=250)
    returns = []
    t0 = time.time()
    for i in range(steps // chunk_steps):
        carry, (info, m) = chunk(carry)
        n, total = float(info["episodes"].sum()), float(info["return"].sum())
        if n:
            returns.append((i, total / n, n))
        if log and (i + 1) % 4 == 0:
            recent = [r for _, r, _ in returns[-4:]]
            log(f"step {(i + 1) * chunk_steps * slots:>7,} transitions  return {np.mean(recent) if recent else float('nan'):8.1f}  alpha {float(m['alpha'][-1]):.3f}  {time.time() - t0:5.0f} s")
    last = [r for i, r, _ in returns if i >= 0.8 * (steps // chunk_steps)]
    return float(np.mean(last))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check TQC on the pendulum swing-up.")
    ap.add_argument("--steps", type=int, default=6000, help="environment steps (x 4 slots)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ret = check(a.steps, seed=a.seed)
    ok = ret > -250
    print(f"average return over the last fifth: {ret:.1f} ({'pass' if ok else 'FAIL'}: above -250 expected)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
