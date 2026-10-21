"""
Torch-free driving and designing: the actor's (and the designer's
Q-network's) forward pass in numpy, weights shared between the learner and
the workers through OS shared memory, frozen opponents, and exploration
noise.

Workers never import torch (it costs ~500 MB RAM per process). The learner
flattens the actor into one float32 array and bumps a version counter; each
worker keeps a private copy and re-copies only when the version changed.
"""

import numpy as np


def layer_shapes(obs_dim, action_dim, hidden):
    """The actor's parameters in order (agent.Actor): Linear, ReLU, Linear,
    ReLU, Linear, tanh."""
    return [(hidden, obs_dim), (hidden,), (hidden, hidden), (hidden,), (action_dim, hidden), (action_dim,)]


def designer_shapes(cfg):
    """The designer's Q-network parameters in order (agent.DesignerNet)."""
    return layer_shapes(cfg.designer_obs_dim, cfg.designer_ops, cfg.designer.hidden)


def mlp_act(params, states):
    return np.tanh(mlp_q(params, states))


def mlp_q(params, states):
    """The 3-layer MLP without an output activation (the designer's Q-values)."""
    w1, b1, w2, b2, w3, b3 = params
    x = np.asarray(states, dtype=np.float32)
    x = np.maximum(x @ w1.T + b1, 0.0)
    x = np.maximum(x @ w2.T + b2, 0.0)
    return x @ w3.T + b3


def greedy_ops(params, obs, mask):
    """The designer's best legal operation per team: argmax of its Q-values
    over the operations the mask allows."""
    return np.where(mask, mlp_q(params, obs), -np.inf).argmax(axis=1)


class SharedPolicy:
    def __init__(self, cfg, ctx, shapes=None):
        """shapes: the network's layer shapes (default: the driver's actor)."""
        self.shapes = shapes or layer_shapes(cfg.obs_dim, cfg.action_dim, cfg.td3.hidden)
        self.size = int(sum(np.prod(s) for s in self.shapes))
        self.flat = ctx.RawArray("f", self.size)
        self.version = ctx.Value("l", 0)
        self.lock = ctx.Lock()

    def publish(self, params):
        """Learner side: params - the actor's weight arrays in layer order."""
        flat = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in params])
        assert flat.size == self.size, "actor layout changed - update policy.layer_shapes"
        with self.lock:
            np.frombuffer(self.flat, dtype=np.float32)[:] = flat
            self.version.value += 1


class WorkerPolicy:
    """A worker's private copy of the shared actor."""

    def __init__(self, shared):
        self.shared = shared
        self.version = -1
        self.refresh()

    def refresh(self):
        if self.shared.version.value == self.version:
            return
        with self.shared.lock:
            flat = np.frombuffer(self.shared.flat, dtype=np.float32).copy()
            self.version = self.shared.version.value
        self.params, off = [], 0
        for shape in self.shared.shapes:
            size = int(np.prod(shape))
            self.params.append(flat[off:off + size].reshape(shape))
            off += size

    def act(self, states):
        return mlp_act(self.params, states)


class FrozenPolicy:
    """Fixed weights (an opponent, or a driver in compare.py), from
    agent.read_actor_weights."""

    def __init__(self, params):
        self.params = [np.asarray(p, dtype=np.float32) for p in params]

    def act(self, states):
        return mlp_act(self.params, states)


class ExplorationNoise:
    """One episode's noise: every car its own sigma, log-uniform in
    exploration_sigma, applied as AR(1) noise x = rho x + sqrt(1 - rho^2) N(0, 1).
    A push lasts ~0.25-0.5 s, long enough to try a move; careful and
    adventurous drivers share every grid."""

    def __init__(self, cfg, rng, num_cars):
        lo, hi = np.log(cfg.td3.exploration_sigma)
        self.rng = rng
        self.sigma = np.exp(rng.uniform(lo, hi, size=(num_cars, 1)))
        self.rho = cfg.td3.exploration_correlation
        self.scale = np.sqrt(1.0 - self.rho ** 2)
        self.state = rng.normal(0.0, 1.0, size=(num_cars, cfg.action_dim))

    def __call__(self):
        self.state = self.rho * self.state + self.scale * self.rng.normal(0.0, 1.0, size=self.state.shape)
        return self.sigma * self.state
