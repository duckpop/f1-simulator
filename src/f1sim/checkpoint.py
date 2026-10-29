"""Checkpoints: an agent's arrays in an .npz (no pickle, so loading one never runs code) and a JSON record of where
they came from. Loading needs a template agent of the same shape (`tqc.init` with the same network settings), whose
structure the arrays are poured back into."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def save(path, agent, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = [np.asarray(a) for a in jax.tree.leaves(jax.device_get(agent))]
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta or {})), **{f"a{i}": a for i, a in enumerate(leaves)})
    tmp.replace(path)  # never a half-written checkpoint


def meta(path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return json.loads(str(d["meta"]))


def load(path, template):
    """(agent shaped like `template`, on the device; meta)."""
    with np.load(path, allow_pickle=False) as d:
        info = json.loads(str(d["meta"]))
        leaves = [d[f"a{i}"] for i in range(len(d.files) - 1)]
    t_leaves, treedef = jax.tree.flatten(template)
    if len(leaves) != len(t_leaves) or any(a.shape != b.shape for a, b in zip(leaves, t_leaves)):
        raise ValueError(f"{path}: the saved networks don't match these settings (different layer sizes?)")
    return jax.tree.unflatten(treedef, [jnp.asarray(a, b.dtype) for a, b in zip(leaves, t_leaves)]), info
