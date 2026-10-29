"""Collect and learn on the GPU: many environment steps, each followed by gradient steps, in one jitted call.

The environment is any `env_step(*args, state, action, key) -> (state, obs, Step, info)` over a fixed set of slots
(one per car): action (S, act_dim) in [-1, 1], obs (S, obs_dim), Step (reward, valid, end) per slot for the step just
taken, and info a dict of scalars to log. `args` (the track bank) are passed on every call rather than baked into the
compiled function.

    chunk = make_chunk(env_step, cfg, steps=500, updates=1, warmup=10_000, learn_from=10_000)
    carry, (info, metrics) = chunk(carry, *args)      # info (steps,) per key, metrics (steps,) per key
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import replay as rp
from . import tqc


class Step(NamedTuple):
    reward: jax.Array  # (S,)
    valid: jax.Array  # (S,) the driver was in control before the step
    end: jax.Array  # (S,) replay.CONTINUE / TERMINAL / CUT


class Carry(NamedTuple):
    env: object
    obs: jax.Array  # (S, obs_dim) the next step's inputs
    agent: tqc.TQCState
    replay: rp.Replay
    key: jax.Array
    steps: jax.Array  # environment steps so far


def make_chunk(env_step, cfg: tqc.TQCConfig, steps: int, updates: int = 1, warmup: int = 0, learn_from: int = 0):
    """warmup: environment steps with uniform random actions; learn_from: environment steps before the first update."""
    zeros = {k: jnp.zeros((), jnp.float32) for k in tqc.METRICS}

    def learn(agent, replay, key):
        def one(agent, k):
            k_sample, k_update = jax.random.split(k)
            batch = rp.sample(replay, k_sample, cfg.batch, cfg.n_step, cfg.gamma)
            return tqc.update(agent, cfg, batch, k_update)

        agent, m = jax.lax.scan(one, agent, jax.random.split(key, updates))
        return agent, jax.tree.map(jnp.mean, m)

    def body(args, c: Carry, _):
        key, k_act, k_rand, k_env, k_learn = jax.random.split(c.key, 5)
        policy = tqc.act(c.agent.actor, cfg, c.obs, k_act)
        random = jax.random.uniform(k_rand, policy.shape, minval=-1.0, maxval=1.0)
        action = jnp.where(c.steps < warmup, random, policy)
        env, obs, st, info = env_step(*args, c.env, action, k_env)
        buf = rp.add(c.replay, c.obs, action, st.reward, st.valid, st.end)
        agent, m = jax.lax.cond(c.steps >= learn_from, lambda: learn(c.agent, buf, k_learn), lambda: (c.agent, zeros))
        m = {**m, "learning": (c.steps >= learn_from).astype(jnp.float32)}
        return Carry(env, obs, agent, buf, key, c.steps + 1), (info, m)

    @partial(jax.jit, donate_argnums=0)
    def chunk(carry: Carry, *args):
        return jax.lax.scan(partial(body, args), carry, None, length=steps)

    return chunk
