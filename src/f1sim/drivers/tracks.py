"""Tracks for training and evals: from a pool built by `f1env-tracks` (the real runs), or generated on the spot (small
runs and tests, when there's no pool).

Widths are "normal" (the 12-15 m tracks) or a wide copy for stage A's curriculum ("26", "20"): the pool's copies for
training tracks; eval tracks are widened on the spot (a copy that doesn't fit is left out).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def widened(tracks, width):
    """Copies at a wider width, leaving out those that don't fit (at 26 m only ~2 of the 10 eval tracks do)."""
    w = _width(width)
    if w is None:
        return list(tracks)
    return [c for c in (widen(t, w) for t in tracks) if c is not None]
