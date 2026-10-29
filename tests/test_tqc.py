import math

import jax
import jax.numpy as jnp
import numpy as np

from f1sim import nets, tqc
from f1sim.nets import EncoderSpec


def small(**kw):
    return tqc.TQCConfig(obs_dim=6, act_dim=2, critics=3, quantiles=5, drop=1, actor_hidden=(16,), critic_hidden=(16, 16), batch=8, **kw)


def test_atoms_kept_per_profile():
    keep = lambda d: tqc.TQCConfig(obs_dim=1, act_dim=1, critics=5, quantiles=25, drop=d).kept
    assert (keep(1), keep(2), keep(3), keep(4)) == (120, 115, 110, 105)  # 5, 10, 15, 20 of 125 dropped


def test_truncation_drops_the_most_optimistic_atoms():
    z = jnp.arange(2 * 1 * 3, dtype=jnp.float32).reshape(2, 1, 3)  # critic 0: 0 1 2, critic 1: 3 4 5
    np.testing.assert_array_equal(tqc.truncated_atoms(z, 4), [[0, 1, 2, 3]])
    z = jnp.array([[[5.0, 1.0]], [[3.0, 9.0]]])
    np.testing.assert_array_equal(tqc.truncated_atoms(z, 2), [[1.0, 3.0]])


def test_quantile_loss_is_zero_on_target_and_pulls_towards_it():
    z = jnp.zeros((2, 4, 3))
    assert float(tqc.quantile_huber(z, jnp.zeros((4, 5)))) == 0.0
    grad = jax.grad(lambda z: tqc.quantile_huber(z, jnp.ones((4, 5))))(z)
    assert bool(jnp.all(grad < 0))  # below the target: every quantile is pushed up


def test_quantiles_spread_to_the_target_distribution():
    target = jnp.linspace(-1.0, 1.0, 101)[None]  # one sample, a uniform spread of atoms
    z = jnp.zeros((1, 1, 4))
    for _ in range(3000):
        z = z - 2.0 * jax.grad(lambda z: tqc.quantile_huber(z, target, kappa=0.01))(z)
    np.testing.assert_allclose(np.asarray(z[0, 0]), [-0.75, -0.25, 0.25, 0.75], atol=0.05)  # the 1/8, 3/8 ... quantiles


def test_tanh_gaussian_log_prob():
    params = nets.actor_init(jax.random.PRNGKey(0), EncoderSpec(), 3, 1, (8,))
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, 3))
    a, logp = nets.actor_sample(params, EncoderSpec(), obs, jax.random.PRNGKey(2))
    mean, log_std = nets.actor_dist(params, EncoderSpec(), obs)
    u = jnp.arctanh(jnp.clip(a, -0.999999, 0.999999))
    std = jnp.exp(log_std)
    expect = -0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi) - jnp.log(1 - jnp.tanh(u) ** 2)
    np.testing.assert_allclose(np.asarray(logp), np.asarray(expect.sum(-1)), rtol=1e-3, atol=1e-3)
    assert bool(jnp.all(jnp.abs(a) <= 1.0))


def test_driver_networks_shapes():
    spec = EncoderSpec(lidar_beams=50)
    actor = nets.actor_init(jax.random.PRNGKey(0), spec, 209, 5, (256, 256))
    critics = nets.ensemble_init(jax.random.PRNGKey(1), 5, spec, 209, 5, (512, 512, 512), 25)
    obs = jnp.zeros((7, 209))
    a, logp = nets.actor_sample(actor, spec, obs, jax.random.PRNGKey(2))
    assert a.shape == (7, 5) and logp.shape == (7,)
    assert nets.ensemble(critics, spec, obs, a).shape == (5, 7, 25)
    assert nets.actor_mean(actor, spec, jnp.zeros((3, 4, 209))).shape == (3, 4, 5)


def test_lidar_branch_reads_the_beams_in_order():
    """The conv sees beam k's three readings together: changing beam 0's barrier reading changes the features of the
    first conv window only."""
    spec = EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,))
    params, n = nets.encoder_init(jax.random.PRNGKey(0), spec, 209)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (1, 209))
    f0 = nets.encode(params, spec, obs)
    f1 = nets.encode(params, spec, obs.at[0, 100].add(1.0))  # barrier channel, beam 0
    changed = np.flatnonzero(np.abs(np.asarray(f1 - f0)[0]) > 1e-6)
    assert f0.shape == (1, n) and len(changed) and changed.max() < 4  # window 0's 4 filters (flattened beam-major)


def test_update_moves_networks_alpha_and_target():
    cfg = small(target_entropy=-10.0)
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    k = jax.random.split(jax.random.PRNGKey(1), 4)
    batch = tqc.Batch(jax.random.normal(k[0], (8, 6)), jax.random.uniform(k[1], (8, 2), minval=-1, maxval=1), jnp.ones(8), jnp.full(8, 0.99), jax.random.normal(k[2], (8, 6)))
    new, m = jax.jit(lambda a, b, key: tqc.update(a, cfg, b, key))(agent, batch, k[3])
    assert all(np.isfinite(float(v)) for v in m.values())
    moved = lambda x, y: any(float(jnp.max(jnp.abs(a - b))) > 0 for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y)))
    assert moved(agent.actor, new.actor) and moved(agent.critic, new.critic)
    # the target follows the critic by tau
    for t0, c1, t1 in zip(jax.tree.leaves(agent.target), jax.tree.leaves(new.critic), jax.tree.leaves(new.target)):
        np.testing.assert_allclose(np.asarray(t1), np.asarray((1 - cfg.tau) * t0 + cfg.tau * c1), rtol=1e-5, atol=1e-6)
    # the policy is far more random than H0 = -10 asks: alpha goes down
    assert float(new.log_alpha) < float(agent.log_alpha)
