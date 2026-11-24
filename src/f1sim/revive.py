"""Bringing dead units back. A hidden unit of a ReLU network that outputs 0 for every input gets no gradient, so it
never learns again; over the stages the drivers' last hidden layer went from 94 live units of 256 to 12 this way
(2026-11-16). `revive` finds the units of a network's head that fire on none of a sample of real inputs and makes them
new: fresh incoming weights (as at the start of training) and a zero bias, so they fire again, and zero outgoing
weights, so nothing downstream changes: the network computes exactly what it did, and training can now give the
units a job. A second copy of the same network (the averaged actor, the target critics) takes the same new weights
for the units dead in both, so the two stay in step.

Works on an actor and on a critic ensemble (a leading axis per critic; the head then also takes the action).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import nets


def _hidden(net, spec, obs, action=None):
    """The head's hidden layers' outputs, [(N?, B, H)]; with `action` the net is a critic ensemble."""

    def through(head, x):
        out = []
        for layer in head[:-1]:
            x = jax.nn.relu(x @ layer["w"] + layer["b"])
            out.append(x)
        return out

    if action is None:
        return through(net["head"], nets.encode(net["enc"], spec, obs))
    return jax.vmap(through)(net["head"], nets.critic_features(net, spec, obs, action))


# compiled once per network shape: the weights are arguments, not constants baked into a fresh compile per call
# (a fresh `jax.jit(lambda ...)` closing over the weights kept ~200 MB per critic revive, every eval, 2026-11-21)
@partial(jax.jit, static_argnums=1)
def _fires_actor(net, spec, o):
    return [jnp.any(h > 0, axis=-2) for h in _hidden(net, spec, o)]


@partial(jax.jit, static_argnums=1)
def _fires_critic(net, spec, o, a):
    return [jnp.any(h > 0, axis=-2) for h in _hidden(net, spec, o, a)]


@partial(jax.jit, static_argnums=1)
def _out_actor(net, spec, o):
    return nets.mlp(net["head"], nets.encode(net["enc"], spec, o), bf16=spec.bf16)


@partial(jax.jit, static_argnums=1)
def _out_critic(net, spec, o, a):
    return nets.ensemble(net, spec, o, a)


def _rows(obs, action, i, batch):
    o = jnp.asarray(obs[i : i + batch], jnp.float32)
    return (o,) if action is None else (o, jnp.asarray(action[i : i + batch], jnp.float32))


def firing(net, spec, obs, action=None, batch: int = 4096):
    """Per hidden layer of the head, which units fire on at least one row: [(H,) bool], or [(N, H)] for critics."""
    f = _fires_actor if action is None else _fires_critic
    fires = None
    for i in range(0, len(obs), batch):
        now = [np.asarray(m) for m in f(net, spec, *_rows(obs, action, i, batch))]
        fires = now if fires is None else [a | b for a, b in zip(fires, now)]
    return fires


def output(net, spec, obs, action=None, batch: int = 4096):
    """What the network computes on the rows: the actor's (B, 2 x actions), the critics' (N, B, quantiles)."""
    f = _out_actor if action is None else _out_critic
    parts = [np.asarray(f(net, spec, *_rows(obs, action, i, batch))) for i in range(0, len(obs), batch)]
    return np.concatenate(parts, axis=-2)


def renew(net, dead, seed: int):
    """`net` with the units in `dead` ([(H,)] or [(N, H)] bool per hidden layer of the head) made new."""
    rng = np.random.default_rng(seed)
    head = [{k: np.array(v) for k, v in layer.items()} for layer in jax.device_get(net["head"])]
    for i, mask in enumerate(dead):
        mask = np.asarray(mask, bool)
        w, b, nxt = head[i]["w"], head[i]["b"], head[i + 1]["w"]
        bound = 1.0 / math.sqrt(w.shape[-2])
        fresh = rng.uniform(-bound, bound, w.shape).astype(w.dtype)
        head[i]["w"] = np.where(mask[..., None, :], fresh, w)  # incoming: every weight into a dead unit
        head[i]["b"] = np.where(mask, 0.0, b).astype(b.dtype)
        head[i + 1]["w"] = np.where(mask[..., :, None], 0.0, nxt).astype(nxt.dtype)  # outgoing: silent until trained
    return {**net, "head": [{k: jnp.asarray(v) for k, v in layer.items()} for layer in head]}


def revive(net, spec, obs, action=None, twin=None, seed: int = 0, tolerance: float = 1e-4):
    """(net, twin, report): the head's dead units made new in `net` and, the same way, in `twin` (a second copy of the
    network, or None); a unit counts as dead when it fires on none of the rows in either copy. Nothing is changed if
    what a copy computes on the rows would move by more than `tolerance` (it shouldn't move at all). report: "alive"
    and "revived" per hidden layer (summed over the critics of an ensemble), "units" per layer, "change" (the largest
    change in any output), "refused"."""
    fires = firing(net, spec, obs, action)
    if twin is not None:
        fires = [a | b for a, b in zip(fires, firing(twin, spec, obs, action))]
    dead = [~f for f in fires]
    report = {"alive": [int(f.sum()) for f in fires], "revived": [int(d.sum()) for d in dead], "units": [int(f.size) for f in fires], "change": 0.0, "refused": False}
    if not any(report["revived"]):
        return net, twin, report
    new, new_twin = renew(net, dead, seed), None if twin is None else renew(twin, dead, seed)
    change = float(np.abs(output(new, spec, obs, action) - output(net, spec, obs, action)).max())
    if twin is not None:
        change = max(change, float(np.abs(output(new_twin, spec, obs, action) - output(twin, spec, obs, action)).max()))
    report["change"] = change
    if not change <= tolerance:
        report["refused"] = True
        return net, twin, report
    return new, new_twin, report


def describe(report) -> str:
    parts = [f"{a}/{u}" + (f" (+{r} revived)" if r else "") for a, u, r in zip(report["alive"], report["units"], report["revived"])]
    return "alive " + ", ".join(parts) + (f"; REFUSED: outputs would move {report['change']:.2g}" if report["refused"] else "")
