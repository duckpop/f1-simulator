"""Collect and learn on the GPU: many environment steps, each followed by gradient steps, in one jitted call.

The environment is any `env_step(*args, state, action, key) -> (state, obs, Step, info)` over a fixed set of slots
(one per car): action (S, act_dim) in [-1, 1], obs (S, obs_dim), Step (reward, valid, end) per slot for the step just
taken, and info a dict of scalars to log. `args` (the track bank) are passed on every call rather than baked into the
compiled function.

    chunk = make_chunk(env_step, cfg, steps=500, updates=1, warmup=10_000, learn_from=10_000)
    carry, (info, metrics) = chunk(carry, *args)      # info (steps,) per key, metrics (steps,) per key

When learning starts (`schedule`): a run from scratch learns at once (as soon as a batch can be drawn), its fresh
policy exploring; a run that starts from a checkpoint (a resume, another profile's networks, stage B from stage A) first
fills a share of its replay with the loaded driver's driving, then trains the critics alone for a number of gradient
steps while the driver keeps driving (the actor and alpha wait), and only then the whole agent. A loaded driver
learning from critics that haven't priced this run's rewards and data yet gets wrecked within a few thousand gradient
steps (together6 at step 32k, 2026-11-05; v2 waited for 500k critic steps). After the critics' phase the actor's and
the temperature's steps ramp up linearly from 0 to full over `actor_ramp` gradient steps (v2: 100k), so the loaded
driver changes gradually instead of all at once.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import replay as rp
from . import tqc


class Schedule(NamedTuple):
    learn_from: int  # environment steps before the first gradient step
    actor_from: int  # environment steps before the actor and alpha learn too (the critics alone before)
    full_from: int = 0  # environment steps from which they learn at the full rate (ramped up from actor_from)

    def actor_share(self, steps):
        """0 before actor_from, then rising linearly to 1 at full_from (traced steps)."""
        ramp = max(self.full_from - self.actor_from, 0)
        if ramp == 0:
            return (steps >= self.actor_from).astype(jnp.float32)
        return jnp.clip((steps - self.actor_from + 1) / ramp, 0.0, 1.0)


def schedule(from_checkpoint: bool, rows: int, slots: int, updates: int, batch: int, n_step: int, fill: float = 0.2, critic_steps: int = 20_000, actor_ramp: int = 100_000) -> Schedule:
    """From scratch: learn as soon as a batch can be drawn, everything at once. From a checkpoint: once `fill` of the
    replay's rows are written, then `critic_steps` gradient steps on the critics alone (`updates` per environment
    step) before the actor and alpha learn too, their steps ramping up to full over `actor_ramp` gradient steps."""
    first = -(-batch // max(slots, 1)) + n_step + 1  # rows before the first batch has a valid window
    if not from_checkpoint:
        return Schedule(first, first, first)
    learn_from = max(first, int(round(fill * rows)))
    actor_from = learn_from + -(-int(critic_steps) // max(updates, 1))
    return Schedule(learn_from, actor_from, actor_from + -(-int(actor_ramp) // max(updates, 1)))


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


def _zeros():
    return {k: jnp.zeros((), jnp.float32) for k in tqc.METRICS}


def learn(agent: tqc.TQCState, cfg: tqc.TQCConfig, replay: rp.Replay, key, updates: int, actor_on=True, target_entropy=None):
    """`updates` gradient steps, each on a fresh batch from the replay (skipped if the batch found no valid step); the
    metrics averaged over them. actor_on (traced): False trains the critics only, a share in (0, 1] scales the actor's
    and the temperature's steps (`tqc.update`); target_entropy (traced): the temperature's target, if not cfg's."""

    def one(agent, k):
        k_sample, k_update = jax.random.split(k)
        batch = rp.sample(replay, k_sample, cfg.batch, cfg.n_step, cfg.gamma)
        return jax.lax.cond(batch.ok, lambda: tqc.update(agent, cfg, batch, k_update, actor_on, target_entropy), lambda: (agent, _zeros()))

    agent, m = jax.lax.scan(one, agent, jax.random.split(key, updates))
    return agent, jax.tree.map(jnp.mean, m)


def maybe_learn(on, agent: tqc.TQCState, cfg: tqc.TQCConfig, replay: rp.Replay, key, updates: int, actor_on=True, target_entropy=None):
    """learn() if `on` (traced), else the agent unchanged and zero metrics."""
    return jax.lax.cond(on, lambda: learn(agent, cfg, replay, key, updates, actor_on, target_entropy), lambda: (agent, _zeros()))


def make_chunk(env_step, cfg: tqc.TQCConfig, steps: int, updates: int = 1, warmup: int = 0, learn_from: int = 0, actor_from: int = 0, full_from: int | None = None):
    """warmup: environment steps with uniform random actions; learn_from: environment steps before the first update;
    actor_from: environment steps before the actor and alpha learn too (the critics alone before); full_from: when
    their steps reach full size, ramped up linearly from actor_from (default: at once)."""
    plan = Schedule(learn_from, actor_from, actor_from if full_from is None else full_from)

    def body(args, c: Carry, _):
        key, k_act, k_rand, k_env, k_learn = jax.random.split(c.key, 5)
        policy = tqc.act(c.agent.actor, cfg, c.obs, k_act)
        random = jax.random.uniform(k_rand, policy.shape, minval=-1.0, maxval=1.0)
        action = jnp.where(c.steps < warmup, random, policy)
        env, obs, st, info = env_step(*args, c.env, action, k_env)
        buf = rp.add(c.replay, c.obs, action, st.reward, st.valid, st.end)
        actor_on = plan.actor_share(c.steps)
        agent, m = maybe_learn(c.steps >= learn_from, c.agent, cfg, buf, k_learn, updates, actor_on)
        m = {**m, "learning": (c.steps >= learn_from).astype(jnp.float32), "actor": actor_on}
        return Carry(env, obs, agent, buf, key, c.steps + 1), (info, m)

    @partial(jax.jit, donate_argnums=0)
    def chunk(carry: Carry, *args):
        return jax.lax.scan(partial(body, args), carry, None, length=steps)

    return chunk
