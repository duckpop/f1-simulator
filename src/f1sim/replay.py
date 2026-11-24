"""The replay buffer, on the GPU: a ring over time for a fixed set of slots (one slot per car), float16 inputs, or
packed by a `Codec`: most inputs in 8 bits (each mapped over a fixed range, a square-root scale for distances so the
near ones keep their precision), the sensitive ones in float16, so the same memory holds ~1.8x the steps.

Each step is stored once, (input, action, reward, valid, end), and a sample's next input is found k rows on in the same
slot, so nothing is stored twice. n-step returns are built when sampling:

- `valid`: the driver was in control before this step (not on the grid before the lights, not crashed or finished);
  only valid steps start a sample.
- `end`: CONTINUE; TERMINAL (a crash, a black flag or the finish: no bootstrap after it); or CUT (the race restarted
  after this step under a car still going: the next row belongs to another race, so this step can't be a sample's
  last, and a window reaching it bootstraps from its input instead).

    buf = new(rows, slots, obs_dim, act_dim)               # codec=...: packed inputs
    buf = add(buf, obs, action, reward, valid, end)        # every slot, one step
    batch = sample(buf, key, 256, n=5, gamma=0.995)        # tqc.Batch
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .tqc import Batch

CONTINUE, TERMINAL, CUT = 0, 1, 2


class Codec(NamedTuple):
    """How inputs are packed: input j in `narrow` is stored in 8 bits over [lo[j], hi[j]] (clipped), on a square-root
    scale where `root[j]`; the inputs in `wide` stay float16."""

    lo: jax.Array  # (obs_dim,) float32
    hi: jax.Array
    root: jax.Array  # (obs_dim,) bool
    narrow: jax.Array  # (n8,) int32: the inputs stored in 8 bits
    wide: jax.Array  # (n16,) int32: the inputs stored in float16
    order: jax.Array  # (obs_dim,) int32: where each input is in narrow + wide


def make_codec(spec) -> Codec:
    """spec: per input (lo, hi, root) for 8 bits, or None for float16."""
    narrow = [j for j, x in enumerate(spec) if x is not None]
    wide = [j for j, x in enumerate(spec) if x is None]
    lo = [x[0] if x else 0.0 for x in spec]
    hi = [x[1] if x else 1.0 for x in spec]
    root = [bool(x[2]) if x else False for x in spec]
    order = jnp.argsort(jnp.asarray(narrow + wide, jnp.int32))
    return Codec(jnp.asarray(lo, jnp.float32), jnp.asarray(hi, jnp.float32), jnp.asarray(root), jnp.asarray(narrow, jnp.int32), jnp.asarray(wide, jnp.int32), order.astype(jnp.int32))


class Packed(NamedTuple):
    narrow: jax.Array  # (..., n8) uint8
    wide: jax.Array  # (..., n16) float16


def encode(c: Codec, obs) -> Packed:
    x = obs[..., c.narrow].astype(jnp.float32)
    lo, hi, root = c.lo[c.narrow], c.hi[c.narrow], c.root[c.narrow]
    t = jnp.clip((x - lo) / (hi - lo), 0.0, 1.0)
    t = jnp.where(root, jnp.sqrt(t), t)
    return Packed(jnp.round(t * 255.0).astype(jnp.uint8), obs[..., c.wide].astype(jnp.float16))


def decode(c: Codec, p: Packed):
    """The inputs (..., obs_dim) float32."""
    lo, hi, root = c.lo[c.narrow], c.hi[c.narrow], c.root[c.narrow]
    t = p.narrow.astype(jnp.float32) / 255.0
    x = lo + jnp.where(root, t * t, t) * (hi - lo)
    return jnp.concatenate([x, p.wide.astype(jnp.float32)], axis=-1)[..., c.order]


class Replay(NamedTuple):
    obs: object  # (T, S, obs_dim) float16, or a Packed of (T, S, n8) uint8 and (T, S, n16) float16
    action: jax.Array  # (T, S, act_dim) float16
    reward: jax.Array  # (T, S)
    valid: jax.Array  # (T, S) bool
    end: jax.Array  # (T, S) int8
    ptr: jax.Array  # next row to write
    size: jax.Array  # rows written (up to T)
    codec: Codec | None = None


def new(rows: int, slots: int, obs_dim: int, act_dim: int, codec: Codec | None = None) -> Replay:
    z = jnp.zeros((rows, slots))
    if codec is None:
        obs = jnp.zeros((rows, slots, obs_dim), jnp.float16)
    else:
        codec = jax.tree.map(jnp.copy, codec)  # its own copy: a chunk donates its replays, and two can't hand over the same arrays
        obs = Packed(jnp.zeros((rows, slots, codec.narrow.shape[0]), jnp.uint8), jnp.zeros((rows, slots, codec.wide.shape[0]), jnp.float16))
    return Replay(
        obs,
        jnp.zeros((rows, slots, act_dim), jnp.float16),
        z,
        z.astype(bool),
        z.astype(jnp.int8),
        jnp.zeros((), jnp.int32),
        jnp.zeros((), jnp.int32),
        codec,
    )


def bytes_per_step(obs_dim: int, act_dim: int, codec: Codec | None = None) -> int:
    """GPU memory per stored step: inputs, action (float16), reward (float32), valid and end (a byte each)."""
    inputs = 2 * obs_dim if codec is None else int(codec.narrow.shape[0]) + 2 * int(codec.wide.shape[0])
    return inputs + 2 * act_dim + 4 + 2


def rows_for(transitions: int, slots: int) -> int:
    """Rows for a buffer of about `transitions` (at least 64)."""
    return max(64, transitions // max(slots, 1))


def add(buf: Replay, obs, action, reward, valid, end) -> Replay:
    i = buf.ptr
    put = lambda a, x: jax.lax.dynamic_update_index_in_dim(a, x.astype(a.dtype), i, axis=0)
    rows = buf.reward.shape[0]
    if buf.codec is None:
        stored = put(buf.obs, obs)
    else:
        p = encode(buf.codec, obs)
        stored = Packed(put(buf.obs.narrow, p.narrow), put(buf.obs.wide, p.wide))
    return buf._replace(
        obs=stored,
        action=put(buf.action, action),
        reward=put(buf.reward, reward),
        valid=put(buf.valid, valid),
        end=put(buf.end, end),
        ptr=(i + 1) % rows,
        size=jnp.minimum(buf.size + 1, rows),
    )


def clear(buf: Replay) -> Replay:
    """The same buffer with nothing in it (its rows are written over as it fills again)."""
    return buf._replace(ptr=jnp.zeros((), jnp.int32), size=jnp.zeros((), jnp.int32))


def inputs(buf: Replay, t, s):
    """The stored inputs at rows t, slots s (float16, or float32 decoded from a packed buffer)."""
    if buf.codec is None:
        return buf.obs[t, s]
    return decode(buf.codec, Packed(buf.obs.narrow[t, s], buf.obs.wide[t, s]))


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


def sample(buf: Replay, key, batch: int, n: int, gamma: float, candidates: int = 16) -> Batch:
    """A batch of n-step transitions, uniform over valid starting steps (drawn from `candidates` x batch tries, so with
    a quarter of the rows valid a batch still has few repeats). `ok` is False if no try found a valid step: the batch
    is then meaningless and the learner skips it (`loop.learn`)."""
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
    return Batch(inputs(buf, t0, s), buf.action[t0, s], ret, discount, inputs(buf, t_next, s), total > 0)


def cut_last(buf: Replay, mask) -> Replay:
    """Mark the last step written as CUT for the slots in mask (S,): their sessions were restarted from outside the
    environment step (a track swap), so the next row isn't a continuation."""
    i = (buf.ptr - 1) % buf.reward.shape[0]
    end = buf.end[i]
    end = jnp.where(mask & buf.valid[i] & (end == CONTINUE), CUT, end).astype(end.dtype)
    return buf._replace(end=buf.end.at[i].set(end))
