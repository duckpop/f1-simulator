"""The replay buffer, on the GPU: a ring over time for a fixed set of slots (one slot per car), float16 inputs.

Each step is stored once, (input, action, reward, valid, end), and a sample's next input is found k rows on in the same
slot, so nothing is stored twice. n-step returns are built when sampling:

- `valid`: the driver was in control before this step (not on the grid before the lights, not crashed or finished);
  only valid steps start a sample.
- `end`: CONTINUE; TERMINAL (a crash, a black flag or the finish: no bootstrap after it); or CUT (the race restarted
  after this step under a car still going: the next row belongs to another race, so this step can't be a sample's
  last, and a window reaching it bootstraps from its input instead).

    buf = new(rows, slots, obs_dim, act_dim)
    buf = add(buf, obs, action, reward, valid, end)        # every slot, one step
    batch = sample(buf, key, 256, n=5, gamma=0.995)        # tqc.Batch
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .tqc import Batch

CONTINUE, TERMINAL, CUT = 0, 1, 2


class Replay(NamedTuple):
    obs: jax.Array  # (T, S, obs_dim) float16
    action: jax.Array  # (T, S, act_dim) float16
    reward: jax.Array  # (T, S)
    valid: jax.Array  # (T, S) bool
    end: jax.Array  # (T, S) int8
    ptr: jax.Array  # next row to write
    size: jax.Array  # rows written (up to T)


def new(rows: int, slots: int, obs_dim: int, act_dim: int) -> Replay:
    z = jnp.zeros((rows, slots))
    return Replay(
        jnp.zeros((rows, slots, obs_dim), jnp.float16),
        jnp.zeros((rows, slots, act_dim), jnp.float16),
        z,
        z.astype(bool),
        z.astype(jnp.int8),
        jnp.zeros((), jnp.int32),
        jnp.zeros((), jnp.int32),
    )


def rows_for(transitions: int, slots: int) -> int:
    """Rows for a buffer of about `transitions` (at least 64)."""
    return max(64, transitions // max(slots, 1))


def add(buf: Replay, obs, action, reward, valid, end) -> Replay:
    i = buf.ptr
    put = lambda a, x: jax.lax.dynamic_update_index_in_dim(a, x.astype(a.dtype), i, axis=0)
    rows = buf.reward.shape[0]
    return Replay(
        put(buf.obs, obs),
        put(buf.action, action),
        put(buf.reward, reward),
        put(buf.valid, valid),
        put(buf.end, end),
        (i + 1) % rows,
        jnp.minimum(buf.size + 1, rows),
    )


def nstep(reward, end, valid, n: int, gamma: float):
    """For windows of n + 1 rows starting at a valid step (each (B, n + 1)): (return, discount, row to bootstrap from).

    The window stops at the first row k >= 1 that's CUT or not valid, or at row n; the steps before it count, up to and
    including a terminal. It bootstraps from row k with gamma^k, or not at all after a terminal or at an invalid row."""
    ks = jnp.arange(n + 1)
    stop = ((end == CUT) | ~valid) & (ks >= 1)
    stop = stop.at[:, n].set(True)
    k_stop = jnp.argmax(stop, axis=1)
    term = end == TERMINAL
    term_before = (jnp.cumsum(term, axis=1) - term) > 0  # a terminal strictly before row k
    counts = (ks[None] < k_stop[:, None]) & ~term_before
    ret = jnp.sum(jnp.where(counts, gamma ** ks[None] * reward, 0.0), axis=1)
    ended = jnp.any(term & (ks[None] < k_stop[:, None]), axis=1)
    boot_ok = jnp.take_along_axis(valid, k_stop[:, None], axis=1)[:, 0]
    discount = jnp.where(ended | ~boot_ok, 0.0, gamma**k_stop)
    return ret, discount, k_stop


def sample(buf: Replay, key, batch: int, n: int, gamma: float, candidates: int = 4) -> Batch:
    """A batch of n-step transitions, uniform over valid starting steps (drawn from `candidates` x batch tries)."""
    rows, slots = buf.reward.shape
    k_row, k_slot, k_pick = jax.random.split(key, 3)
    tries = batch * candidates
    offset = jax.random.randint(k_row, (tries,), 0, jnp.maximum(buf.size - n, 1))  # the window [o, o + n] is written
    t0 = (buf.ptr - buf.size + offset) % rows
    s = jax.random.randint(k_slot, (tries,), 0, slots)
    ok = buf.valid[t0, s] & (buf.end[t0, s] != CUT)
    total = jnp.sum(ok)
    p = jnp.where(total > 0, ok / jnp.maximum(total, 1), 1.0 / tries)
    pick = jax.random.choice(k_pick, tries, (batch,), p=p)
    t0, s = t0[pick], s[pick]
    idx = (t0[:, None] + jnp.arange(n + 1)[None]) % rows  # (B, n + 1)
    ret, discount, k = nstep(buf.reward[idx, s[:, None]], buf.end[idx, s[:, None]], buf.valid[idx, s[:, None]], n, gamma)
    t_next = jnp.take_along_axis(idx, k[:, None], axis=1)[:, 0]
    return Batch(buf.obs[t0, s], buf.action[t0, s], ret, discount, buf.obs[t_next, s])
