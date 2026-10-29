"""Networks in plain JAX. Parameters are nested dicts and lists of arrays, so an ensemble of critics is one vmap away.

The driver's inputs are split in two, in the order f1env's DRIVER_LAYOUT gives them: the lidar (50 beams x 3
readings: edge, closing speed, barrier) goes through a 1D convolution along the beams; the other ~59 inputs (car,
tyres, race, orders, status) go through a small MLP. The two are joined and fed to the head: the actor's or a
critic's. Every network has its own encoder, so the critics' and the actor's losses never fight over shared weights.
For small test tasks (`lidar_beams = 0`) the encoder is the identity.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class EncoderSpec(NamedTuple):
    lidar_beams: int = 0  # 0: no lidar branch, the inputs go straight to the head
    lidar_channels: int = 3
    conv: tuple = ((32, 5, 2), (64, 3, 2))  # (filters, width, stride) per layer
    other_hidden: tuple = (128,)


def _uniform(key, shape, bound):
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


def dense_init(key, n_in: int, n_out: int):
    kw, kb = jax.random.split(key)
    bound = 1.0 / math.sqrt(n_in)
    return {"w": _uniform(kw, (n_in, n_out), bound), "b": _uniform(kb, (n_out,), bound)}


def mlp_init(key, sizes):
    """sizes: [inputs, hidden..., outputs]."""
    keys = jax.random.split(key, len(sizes) - 1)
    return [dense_init(k, a, b) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def mlp(layers, x, last_relu: bool = False):
    for i, p in enumerate(layers):
        x = x @ p["w"] + p["b"]
        if i < len(layers) - 1 or last_relu:
            x = jax.nn.relu(x)
    return x


def _conv_init(key, c_in: int, filters: int, width: int):
    kw, kb = jax.random.split(key)
    bound = 1.0 / math.sqrt(c_in * width)
    return {"w": _uniform(kw, (width, c_in, filters), bound), "b": _uniform(kb, (filters,), bound)}


def encoder_init(key, spec: EncoderSpec, obs_dim: int):
    """(params, features out)."""
    if spec.lidar_beams == 0:
        return {}, obs_dim
    k_conv, k_other = jax.random.split(key)
    keys = jax.random.split(k_conv, len(spec.conv))
    convs, width, channels = [], spec.lidar_beams, spec.lidar_channels
    for k, (filters, w, stride) in zip(keys, spec.conv):
        convs.append(_conv_init(k, channels, filters, w))
        width, channels = (width - w) // stride + 1, filters
    n_other = obs_dim - spec.lidar_beams * spec.lidar_channels
    other = mlp_init(k_other, [n_other, *spec.other_hidden])
    out = width * channels + (spec.other_hidden[-1] if spec.other_hidden else n_other)
    return {"conv": convs, "other": other}, out


def encode(params, spec: EncoderSpec, obs):
    if spec.lidar_beams == 0:
        return obs
    lead = obs.shape[:-1]
    n = spec.lidar_beams * spec.lidar_channels
    # the observation holds the lidar channel by channel (all edges, then closing speeds, then barriers)
    x = obs[..., :n].reshape(-1, spec.lidar_channels, spec.lidar_beams).transpose(0, 2, 1)
    for p, (_, _, stride) in zip(params["conv"], spec.conv):
        x = jax.lax.conv_general_dilated(x, p["w"], (stride,), "VALID", dimension_numbers=("NWC", "WIO", "NWC"))
        x = jax.nn.relu(x + p["b"])
    lidar = x.reshape(lead + (-1,))
    other = mlp(params["other"], obs[..., n:], last_relu=True) if spec.other_hidden else obs[..., n:]
    return jnp.concatenate([lidar, other], axis=-1)


# ---- the actor: a tanh-squashed Gaussian -------------------------------------------------------------------------


def actor_init(key, spec: EncoderSpec, obs_dim: int, act_dim: int, hidden):
    k_enc, k_head = jax.random.split(key)
    enc, n = encoder_init(k_enc, spec, obs_dim)
    return {"enc": enc, "head": mlp_init(k_head, [n, *hidden, 2 * act_dim])}


def actor_dist(params, spec: EncoderSpec, obs):
    """(mean, log std) of the Gaussian before the tanh."""
    out = mlp(params["head"], encode(params["enc"], spec, obs))
    mean, log_std = jnp.split(out, 2, axis=-1)
    return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


def actor_sample(params, spec: EncoderSpec, obs, key):
    """(action in [-1, 1], its log-probability)."""
    mean, log_std = actor_dist(params, spec, obs)
    eps = jax.random.normal(key, mean.shape)
    u = mean + jnp.exp(log_std) * eps
    log_normal = -0.5 * eps**2 - log_std - 0.5 * math.log(2 * math.pi)
    log_tanh_grad = 2.0 * (math.log(2.0) - u - jax.nn.softplus(-2.0 * u))  # log(1 - tanh(u)^2), stable
    return jnp.tanh(u), jnp.sum(log_normal - log_tanh_grad, axis=-1)


def actor_mean(params, spec: EncoderSpec, obs):
    """The deterministic action (evals)."""
    return jnp.tanh(actor_dist(params, spec, obs)[0])


# ---- the critics: an ensemble of quantile networks ---------------------------------------------------------------


def critic_init(key, spec: EncoderSpec, obs_dim: int, act_dim: int, hidden, quantiles: int):
    k_enc, k_head = jax.random.split(key)
    enc, n = encoder_init(k_enc, spec, obs_dim)
    return {"enc": enc, "head": mlp_init(k_head, [n + act_dim, *hidden, quantiles])}


def ensemble_init(key, n: int, spec: EncoderSpec, obs_dim: int, act_dim: int, hidden, quantiles: int):
    """n critics' parameters, stacked on a leading axis."""
    return jax.vmap(lambda k: critic_init(k, spec, obs_dim, act_dim, hidden, quantiles))(jax.random.split(key, n))


def ensemble(params, spec: EncoderSpec, obs, action):
    """(n critics, ..., quantiles)."""

    def one(p):
        return mlp(p["head"], jnp.concatenate([encode(p["enc"], spec, obs), action], axis=-1))

    return jax.vmap(one)(params)
