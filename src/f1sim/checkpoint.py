"""Checkpoints: an agent's arrays in an .npz (no pickle, so loading one never runs code) and a JSON record of where
they came from. Loading needs a template agent of the same shape (`tqc.init` with the same network settings), whose
structure the arrays are poured back into; given the settings, a checkpoint with fewer hidden layers loads too, deepened
with identity layers (the same driver)."""

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


def _deepen(head, extra: int):
    """An MLP head with `extra` identity layers just before its output layer. The layer before ends in a ReLU, so its
    output is never negative and ReLU(I x + 0) = x: the head computes exactly what it did. Works on a critic
    ensemble's stacked heads too (a leading axis per critic)."""
    w = head[-1]["w"]
    lead, h = w.shape[:-2], w.shape[-2]
    eye = {"w": jnp.broadcast_to(jnp.eye(h, dtype=w.dtype), lead + (h, h)), "b": jnp.zeros(lead + (h,), w.dtype)}
    return head[:-1] + [eye] * extra + head[-1:]


def _widen(net, extra: int):
    """A network whose other-inputs layer takes `extra` more inputs at the end, with zero weights: it computes exactly
    what it did, whatever they are."""
    if not extra:
        return net
    first = net["enc"]["other"][0]
    w = first["w"]
    rows = jnp.zeros(w.shape[:-2] + (extra, w.shape[-1]), w.dtype)
    other = [{**first, "w": jnp.concatenate([w, rows], axis=-2)}] + net["enc"]["other"][1:]
    return {**net, "enc": {**net["enc"], "other": other}}


def _inputs_saved(leaves, template, cfg):
    """How many inputs a checkpoint's networks took: the actor's other-inputs layer (the same leaf at any depth, as the
    actor and its encoder come first) has one row per input after the lidar."""
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(template)[0]]
    i = next(k for k, p in enumerate(paths) if p.endswith("actor['enc']['other'][0]['w']"))
    return cfg.obs_dim - (jax.tree.leaves(template)[i].shape[0] - leaves[i].shape[0])


def load(path, template, cfg=None):
    """(agent shaped like `template`, on the device; meta). With `cfg`, the template's TQCConfig, an older checkpoint
    loads too: one saved with fewer inputs (inputs appended since) gets them with zero weights (`_widen`), one with
    fewer hidden layers of the same width gets them as identities just before each head's output (`_deepen`): the same
    driver and critics, ready to learn with the new inputs and depth; with fresh optimisers."""
    with np.load(path, allow_pickle=False) as d:
        info = json.loads(str(d["meta"]))
        leaves = [d[f"a{i}"] for i in range(len(d.files) - 1)]
    agent = _fits(leaves, template)
    if agent is not None:
        return agent, info
    if cfg is not None and cfg.encoder.lidar_beams:
        from . import tqc

        n_obs = _inputs_saved(leaves, template, cfg)
        a, c = cfg.actor_hidden, cfg.critic_hidden
        for da in range(len(a)):
            for dc in range(len(c)):
                if (da or dc or n_obs != cfg.obs_dim) and len(set(a[len(a) - da - 1 :])) == 1 and len(set(c[len(c) - dc - 1 :])) == 1:  # identities need equal widths
                    small = cfg._replace(obs_dim=n_obs, actor_hidden=a[: len(a) - da], critic_hidden=c[: len(c) - dc])
                    old = _fits(leaves, tqc.init(jax.random.PRNGKey(0), small))
                    if old is not None:
                        grow = lambda net, n: _widen({**net, "head": _deepen(net["head"], n)} if n else net, cfg.obs_dim - n_obs)
                        agent = tqc.from_networks(cfg, grow(old.actor, da), grow(old.critic, dc), grow(old.target, dc), old.log_alpha)
                        return agent, {**info, "deepened": [da, dc], "widened": cfg.obs_dim - n_obs}
    raise ValueError(f"{path}: the saved networks don't match these settings (different layer sizes?)")
