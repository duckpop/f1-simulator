"""Checkpoints: an agent's arrays in an .npz (no pickle, so loading one never runs code) and a JSON record of where
they came from. Loading needs a template agent of the same shape (`tqc.init` with the same network settings), whose
structure the arrays are poured back into. (Training restarted from scratch on 2026-11-24: the migrations of older
checkpoint layouts are gone.)"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


REPO = Path(__file__).resolve().parents[2]


def provenance(repo=REPO) -> dict:
    """Where a run's code came from: the git commit, whether the code or configs had uncommitted changes, and a hash of
    the configs (so a checkpoint names exactly what trained it)."""
    repo = Path(repo)
    out: dict = {}
    try:
        git = lambda *args: subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=True).stdout
        out["git"] = git("rev-parse", "HEAD").strip()
        out["dirty"] = bool(git("status", "--porcelain", "--", "src", "configs", "pyproject.toml").strip())
    except (OSError, subprocess.SubprocessError):
        out["git"] = None
    h = hashlib.sha256()
    for f in sorted((repo / "configs").rglob("*.toml")):
        h.update(f.relative_to(repo).as_posix().encode() + b"\0" + f.read_bytes())
    out["config_sha256"] = h.hexdigest()[:16]
    return out


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


def best_bar(path, **same) -> dict | None:
    """The eval result an existing best checkpoint was saved for, as the bar a new run has to beat, if its eval was the
    same kind (every key in `same` matches its record); otherwise None (a new run's first eval sets the bar)."""
    path = Path(path)
    if not path.exists():
        return None
    info = meta(path)
    return info.get("eval") if all(info.get(k) == v for k, v in same.items()) else None


def _fits(leaves, template):
    t_leaves, treedef = jax.tree.flatten(template)
    if len(leaves) != len(t_leaves) or any(a.shape != b.shape for a, b in zip(leaves, t_leaves)):
        return None
    return jax.tree.unflatten(treedef, [jnp.asarray(a, b.dtype) for a, b in zip(leaves, t_leaves)])


def load(path, template):
    """(agent shaped like `template`, on the device; meta). The networks must match the template's settings."""
    with np.load(path, allow_pickle=False) as d:
        info = json.loads(str(d["meta"]))
        leaves = [d[f"a{i}"] for i in range(len(d.files) - 1)]
    agent = _fits(leaves, template)
    if agent is None:
        raise ValueError(f"{path}: the saved networks don't match these settings (different layer sizes, or critics with a shared encoder?)")
    return agent, info
