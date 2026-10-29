"""Truncated Quantile Critics (Kuznetsov et al., 2020) in JAX.

SAC's stochastic actor and automatic entropy tuning, with an ensemble of N distributional critics of M quantiles
each. The target pools all N x M atoms of the target critics at the next state, sorts them and drops the top d x N
(`drop` = d, per critic): dropping more makes the critic more pessimistic, so the policy avoids risk.

    cfg = TQCConfig(obs_dim=209, act_dim=5, encoder=EncoderSpec(lidar_beams=50))
    agent = init(key, cfg)
    action = act(agent.actor, cfg, obs, key)                  # sampled (training) ...
    action = act(agent.actor, cfg, obs, key, mean=True)       # ... or the mean (evals)
    agent, metrics = update(agent, cfg, batch, key)           # one gradient step on the critics, actor and alpha
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from . import nets
from .nets import EncoderSpec

METRICS = ("critic_loss", "actor_loss", "alpha", "entropy", "q")


class TQCConfig(NamedTuple):
    obs_dim: int
    act_dim: int
    encoder: EncoderSpec = EncoderSpec()
    critics: int = 5  # N
    quantiles: int = 25  # M
    drop: int = 2  # d: atoms dropped per critic, so d x N of the N x M pooled atoms
    gamma: float = 0.995
    n_step: int = 5
    target_entropy: float = -2.0  # H0
    lr: float = 3e-4
    batch: int = 256
    tau: float = 0.005
    actor_hidden: tuple = (256, 256)
    critic_hidden: tuple = (512, 512, 512)
    kappa: float = 1.0  # quantile Huber loss
    init_alpha: float = 1.0

    @property
    def kept(self) -> int:
        return self.critics * self.quantiles - self.drop * self.critics


class TQCState(NamedTuple):
    actor: dict
    critic: dict  # N critics stacked
    target: dict
    log_alpha: jax.Array
    actor_opt: optax.OptState
    critic_opt: optax.OptState
    alpha_opt: optax.OptState
    updates: jax.Array


class Batch(NamedTuple):
    obs: jax.Array  # (B, obs_dim)
    action: jax.Array  # (B, act_dim), in [-1, 1]
    reward: jax.Array  # (B,) the discounted n-step sum
    discount: jax.Array  # (B,) gamma^k for the bootstrap, 0 after a terminal
    next_obs: jax.Array  # (B, obs_dim) the state k steps on


def _adam(cfg: TQCConfig):
    return optax.adam(cfg.lr)


def init(key, cfg: TQCConfig) -> TQCState:
    k_actor, k_critic = jax.random.split(key)
    actor = nets.actor_init(k_actor, cfg.encoder, cfg.obs_dim, cfg.act_dim, cfg.actor_hidden)
    critic = nets.ensemble_init(k_critic, cfg.critics, cfg.encoder, cfg.obs_dim, cfg.act_dim, cfg.critic_hidden, cfg.quantiles)
    log_alpha = jnp.log(jnp.asarray(cfg.init_alpha, jnp.float32))
    opt = _adam(cfg)
    return TQCState(actor, critic, jax.tree.map(jnp.copy, critic), log_alpha, opt.init(actor), opt.init(critic), opt.init(log_alpha), jnp.zeros((), jnp.int32))


def from_networks(cfg: TQCConfig, actor, critic, target=None, log_alpha=None) -> TQCState:
    """Start from trained networks (fine-tuning a profile from another's): fresh optimisers."""
    target = jax.tree.map(jnp.copy, critic if target is None else target)  # its own buffers (the train loop donates)
    log_alpha = jnp.log(jnp.asarray(cfg.init_alpha, jnp.float32)) if log_alpha is None else jnp.asarray(log_alpha, jnp.float32)
    opt = _adam(cfg)
    return TQCState(actor, critic, target, log_alpha, opt.init(actor), opt.init(critic), opt.init(log_alpha), jnp.zeros((), jnp.int32))


def act(actor, cfg: TQCConfig, obs, key, mean: bool = False):
    obs = obs.astype(jnp.float32)
    if mean:
        return nets.actor_mean(actor, cfg.encoder, obs)
    return nets.actor_sample(actor, cfg.encoder, obs, key)[0]


def truncated_atoms(z, kept: int):
    """z (N, B, M) -> (B, kept): every critic's atoms pooled per sample, sorted, the top ones dropped."""
    n, b, m = z.shape
    pooled = jnp.transpose(z, (1, 0, 2)).reshape(b, n * m)
    return jnp.sort(pooled, axis=-1)[:, :kept]


def quantile_huber(z, target, kappa: float = 1.0):
    """z (N, B, M) each critic's quantiles; target (B, K) atoms. The mean over critics, samples, quantiles and atoms."""
    m = z.shape[-1]
    tau = (jnp.arange(m, dtype=jnp.float32) + 0.5) / m
    delta = target[None, :, None, :] - z[..., None]  # (N, B, M, K)
    abs_d = jnp.abs(delta)
    huber = jnp.where(abs_d <= kappa, 0.5 * delta**2, kappa * (abs_d - 0.5 * kappa))
    weight = jnp.abs(tau[None, None, :, None] - (delta < 0).astype(jnp.float32))
    return jnp.mean(weight * huber / kappa)


def update(s: TQCState, cfg: TQCConfig, batch: Batch, key):
    """One gradient step on the critics, then the actor, then the temperature; then the target critics (Polyak)."""
    k_next, k_now = jax.random.split(key)
    enc = cfg.encoder
    obs, next_obs = batch.obs.astype(jnp.float32), batch.next_obs.astype(jnp.float32)
    action = batch.action.astype(jnp.float32)
    alpha = jnp.exp(s.log_alpha)
    opt = _adam(cfg)

    # critics: the truncated, entropy-regularised target
    a_next, logp_next = nets.actor_sample(s.actor, enc, next_obs, k_next)
    z_next = truncated_atoms(nets.ensemble(s.target, enc, next_obs, a_next), cfg.kept)
    target = batch.reward[:, None] + batch.discount[:, None] * (z_next - alpha * logp_next[:, None])
    target = jax.lax.stop_gradient(target)

    def critic_loss(p):
        return quantile_huber(nets.ensemble(p, enc, obs, action), target, cfg.kappa)

    c_loss, grads = jax.value_and_grad(critic_loss)(s.critic)
    step, critic_opt = opt.update(grads, s.critic_opt, s.critic)
    critic = optax.apply_updates(s.critic, step)

    # actor: the mean over every critic's every quantile, less the entropy term
    def actor_loss(p):
        a, logp = nets.actor_sample(p, enc, obs, k_now)
        q = jnp.mean(nets.ensemble(critic, enc, obs, a), axis=(0, 2))
        return jnp.mean(alpha * logp - q), (logp, q)

    (a_loss, (logp, q)), grads = jax.value_and_grad(actor_loss, has_aux=True)(s.actor)
    step, actor_opt = opt.update(grads, s.actor_opt, s.actor)
    actor = optax.apply_updates(s.actor, step)

    # temperature: towards the target entropy
    gap = jax.lax.stop_gradient(jnp.mean(logp) + cfg.target_entropy)
    grad = -gap  # d/d(log alpha) of -log_alpha * gap
    step, alpha_opt = opt.update(grad, s.alpha_opt, s.log_alpha)
    log_alpha = optax.apply_updates(s.log_alpha, step)

    target_net = jax.tree.map(lambda t, c: (1.0 - cfg.tau) * t + cfg.tau * c, s.target, critic)
    metrics = {"critic_loss": c_loss, "actor_loss": a_loss, "alpha": alpha, "entropy": -jnp.mean(logp), "q": jnp.mean(q)}
    return TQCState(actor, critic, target_net, log_alpha, actor_opt, critic_opt, alpha_opt, s.updates + 1), metrics
