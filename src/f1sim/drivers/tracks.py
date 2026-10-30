"""Tracks for training and evals: from a pool built by `f1env-tracks` (the real runs), or generated on the spot (small
runs and tests, when there's no pool).

Widths are "normal" (the 12-15 m tracks) or a wide copy for stage A's curriculum ("26", "20"): the pool's copies for
training tracks; eval tracks are widened on the spot (a copy that doesn't fit is left out).

`Rotation` keeps two sets of training tracks on the GPU and swaps in a fresh set every so often, so a run works through
the whole pool though the GPU holds only a few dozen tracks.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from f1env.sensors import TrackBank, make_bank
from f1env.tracks import generatemap, load_track, widen
from f1env.tracks.pool import EVAL_SEED0, load_pool

EVAL_TRACKS = 10


def _width(width) -> float | None:
    return None if str(width) == "normal" else float(width)


def training_tracks(pool, width="normal", n: int = 16, seed: int = 0, log=print):
    """n training tracks at a width, a random choice from the pool (or seeds 0.. generated, without a pool)."""
    w = _width(width)
    if pool is None:
        out, s = [], 0
        while len(out) < n:
            t = generatemap(seed=s)
            s += 1
            t = t if w is None else widen(t, w)
            if t is not None:
                out.append(t)
        return out
    entries = load_pool(pool)["train"]
    if w is not None:
        entries = [e for e in entries if f"{w:g}" in e["wide"]]
    pick = np.random.default_rng(seed).choice(len(entries), size=min(n, len(entries)), replace=False)
    files = [entries[i]["file"] if w is None else entries[i]["wide"][f"{w:g}"] for i in sorted(pick)]
    if log:
        log(f"loading {len(files)} of {len(entries)} training tracks at width {width}")
    return [load_track(Path(pool) / f) for f in files]


def eval_tracks(pool, width="normal", n: int = EVAL_TRACKS):
    """The fixed eval tracks (the first n), widened when asked."""
    if pool is None:
        tracks = [generatemap(seed=EVAL_SEED0 + k) for k in range(n)]
    else:
        tracks = [load_track(Path(pool) / e["file"]) for e in load_pool(pool)["eval"][:n]]
    return widened(tracks, width)


def eval_track(pool, k: int = 0):
    """Eval track k (0-9) at its normal width."""
    if pool is None:
        return generatemap(seed=EVAL_SEED0 + k)
    return load_track(Path(pool) / load_pool(pool)["eval"][k]["file"])


def widened(tracks, width):
    """Copies at a wider width, leaving out those that don't fit (at 26 m only ~2 of the 10 eval tracks do)."""
    w = _width(width)
    if w is None:
        return list(tracks)
    return [c for c in (widen(t, w) for t in tracks) if c is not None]


# ---- track rotation -----------------------------------------------------------------------------------------------

LINE_FIELDS = ("line_x", "line_y", "line_heading", "half_left", "half_right", "sc_speed", "section", "next_turn", "to_brake")  # (T, samples)
FIELD_GRIDS = ("edge", "barrier", "surface")  # every track's grid, joined


def bank_sizes(bank: TrackBank) -> tuple:
    """(grid cells, centreline samples, pit lane points): the bank's sizes that depend on its tracks."""
    return int(bank.edge.shape[0]), int(bank.line_x.shape[1]), int(bank.pit_lane.shape[1])


def pad_bank(bank: TrackBank, sizes: tuple) -> TrackBank:
    """The bank padded to fixed sizes, so a different set of tracks keeps every array's shape (and the compiled training
    step). Padding is never read: grids are reached through each track's offset, samples and lane points through
    their counts (make_bank already pads shorter tracks the same way)."""
    cells, line, lanes = sizes

    def pad(a, axis, size, mode):
        extra = size - a.shape[axis]
        if extra < 0:
            raise ValueError(f"bank array larger than its padded size ({a.shape[axis]} > {size})")
        width = [(0, 0)] * a.ndim
        width[axis] = (0, extra)
        return a if extra == 0 else jnp.pad(a, width, mode=mode)

    grids = {k: pad(getattr(bank, k), 0, cells, "constant") for k in FIELD_GRIDS}
    lines = {k: pad(getattr(bank, k), 1, line, "edge") for k in LINE_FIELDS}
    return bank._replace(**grids, **lines, pit_lane=pad(bank.pit_lane, 1, lanes, "edge"))


class Rotation:
    """Training tracks for one width, on the GPU in two halves of n tracks: new races draw from the active half. Every
    rotation the other half gets a fresh set and becomes active; the half it replaces was active one rotation ago, so
    every race gets at least a full interval to finish, and only races still on it then are restarted. The pool is
    shuffled once and taken in blocks, so every track comes round before any comes back.

    Without a pool (generated tracks), or with a pool no bigger than n, there's one fixed set and no rotation."""

    # padded sizes leave room for bigger sets than the first, so a swap rarely recompiles: the grids (a sum over the
    # set, ~14 MB a track) vary little; the centreline and pit lane (the longest track's, a few MB) vary more
    HEADROOM = (1.2, 1.5, 1.5)

    def __init__(self, pool, width="normal", n: int = 16, seed: int = 0, generate: bool = False, log=print):
        self.log = log
        self.sizes = None
        self.rotations = 0
        if pool is None or generate:
            self.files, self.rotates = None, False
            self.halves = [training_tracks(pool, width, n, seed, log)]
            self.n = len(self.halves[0])
        else:
            w = _width(width)
            entries = [e for e in load_pool(pool)["train"] if w is None or f"{w:g}" in e["wide"]]
            self.files = [Path(pool) / (e["file"] if w is None else e["wide"][f"{w:g}"]) for e in entries]
            self.n = min(n, len(self.files))
            self.rotates = len(self.files) > self.n
            self.rng = np.random.default_rng(seed)
            self.order, self.taken = [], 0
            self.seen = set()
            self.halves = [self._take(), self._take()] if self.rotates else [self._take()]
            if log:
                log(f"{len(self.files)} training tracks at width {width}: {self.n} at a time" + (f", {2 * self.n} on the GPU (rotating)" if self.rotates else ""))
        self.active = 0
        self.fresh = [False, True]  # the second half is loaded but no race has used it yet
        self.bank = self._build()

    def _take(self):
        """The next n tracks of the shuffled pool (reshuffled when it runs out)."""
        out = []
        while len(out) < self.n:
            if not self.order:
                self.order = list(self.rng.permutation(len(self.files)))
            i = self.order.pop()
            self.seen.add(i)
            out.append(load_track(self.files[i]))
        return out

    def _build(self) -> TrackBank:
        bank = make_bank([t for half in self.halves for t in half])
        sizes = bank_sizes(bank)
        if self.sizes is None or any(s > c for s, c in zip(sizes, self.sizes)):
            grow = sizes if self.sizes is None else tuple(max(s, c) for s, c in zip(sizes, self.sizes))
            self.sizes = tuple(int(np.ceil(g * h)) for g, h in zip(grow, self.HEADROOM))  # a bigger bank: recompiles once
        return pad_bank(bank, self.sizes)

    @property
    def pick(self):
        """(first slot, count) that new races draw from."""
        return jnp.array([self.active * self.n, self.n], jnp.int32)

    def rotate(self):
        """Swap in a fresh set; (bank, the slots replaced: races on them must restart)."""
        if not self.rotates:
            return self.bank, np.zeros(0, np.int32)
        new = 1 - self.active
        replaced = np.zeros(0, np.int32)
        if not self.fresh[new]:
            self.halves[new] = self._take()
            replaced = np.arange(new * self.n, (new + 1) * self.n, dtype=np.int32)
            self.bank = self._build()
        self.fresh[new] = False
        self.active = new
        self.rotations += 1
        return self.bank, replaced

    def describe(self) -> str:
        if not self.rotates:
            return f"{self.n} tracks (fixed)"
        return f"set {self.rotations + 1}: {len(self.seen)} of {len(self.files)} tracks used so far"



def rotate_races(rot: Rotation, state, obs, key, cfg):
    """Swap in the rotation's next set: (bank, env state, inputs (R, C, obs), restarted (R,)). Races on the tracks being
    replaced restart at once, on the new set."""
    from . import env as de

    bank, replaced = rot.rotate()
    state = state._replace(pick=rot.pick)
    hit = np.isin(np.asarray(state.race.tid), replaced)
    if hit.any():
        state, obs = de.compiled("restart_races", cfg)(bank, state, obs, jnp.asarray(hit), key)
    return bank, state, obs, hit
