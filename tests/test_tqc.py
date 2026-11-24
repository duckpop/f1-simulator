import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    grad = jax.grad(lambda z: tqc.quantile_huber(z, target, kappa=0.01))
    z = jax.jit(lambda z: jax.lax.fori_loop(0, 3000, lambda _, z: z - 2.0 * grad(z), z))(jnp.zeros((1, 1, 4)))
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
    """The conv sees beam k's two readings (edge, barrier) together: changing beam 0's barrier reading changes the
    features of the first conv window only."""
    spec = EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,))
    params, n = nets.encoder_init(jax.random.PRNGKey(0), spec, 209)
    obs = jax.random.uniform(jax.random.PRNGKey(1), (1, 209))
    f0 = nets.encode(params, spec, obs)
    f1 = nets.encode(params, spec, obs.at[0, 50].add(1.0))  # barrier channel, beam 0 (all 50 edges come first)
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


def test_a_critic_warm_up_keeps_the_actor_and_alpha_but_moves_their_adam_moments():
    cfg = small(target_entropy=-10.0)
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    k = jax.random.split(jax.random.PRNGKey(1), 4)
    batch = tqc.Batch(jax.random.normal(k[0], (8, 6)), jax.random.uniform(k[1], (8, 2), minval=-1, maxval=1), jnp.ones(8), jnp.full(8, 0.99), jax.random.normal(k[2], (8, 6)))
    new, _ = jax.jit(lambda a, b, key: tqc.update(a, cfg, b, key, actor_on=False))(agent, batch, k[3])
    same = lambda x, y: all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y)))
    assert same(agent.actor, new.actor) and float(new.log_alpha) == float(agent.log_alpha)
    assert not same(agent.critic, new.critic)
    # the actor's and alpha's Adam moments still follow the gradients, ready for the ramp
    assert not same(agent.actor_opt, new.actor_opt) and not same(agent.alpha_opt, new.alpha_opt)


def test_a_ramped_actor_step_is_a_share_of_the_full_one():
    """actor_on as a share (the ramp after the critic warm-up): the actor's and alpha's steps are that share of the
    full ones; the critics' step is the same."""
    cfg = small(target_entropy=-10.0)
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    k = jax.random.split(jax.random.PRNGKey(1), 4)
    batch = tqc.Batch(jax.random.normal(k[0], (8, 6)), jax.random.uniform(k[1], (8, 2), minval=-1, maxval=1), jnp.ones(8), jnp.full(8, 0.99), jax.random.normal(k[2], (8, 6)))
    run = jax.jit(lambda a, b, key, on: tqc.update(a, cfg, b, key, actor_on=on))
    full, _ = run(agent, batch, k[3], 1.0)
    quarter, _ = run(agent, batch, k[3], 0.25)
    for a0, f, q in zip(jax.tree.leaves(agent.actor), jax.tree.leaves(full.actor), jax.tree.leaves(quarter.actor)):
        np.testing.assert_allclose(np.asarray(q - a0), 0.25 * np.asarray(f - a0), rtol=1e-4, atol=1e-7)
    assert float(quarter.log_alpha - agent.log_alpha) == pytest.approx(0.25 * float(full.log_alpha - agent.log_alpha), rel=1e-4)
    for f, q in zip(jax.tree.leaves(full.critic), jax.tree.leaves(quarter.critic)):
        np.testing.assert_array_equal(np.asarray(f), np.asarray(q))


def _one_batch():
    k = jax.random.split(jax.random.PRNGKey(1), 4)
    return tqc.Batch(jax.random.normal(k[0], (8, 6)), jax.random.uniform(k[1], (8, 2), minval=-1, maxval=1), jnp.ones(8), jnp.full(8, 0.99), jax.random.normal(k[2], (8, 6))), k[3]


def test_the_actor_alone_takes_smaller_steps_with_its_own_learning_rate():
    cfg = small(target_entropy=-10.0)
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    batch, key = _one_batch()
    full, _ = tqc.update(agent, cfg, batch, key)
    slow, _ = tqc.update(agent, cfg._replace(actor_lr=cfg.lr / 10), batch, key)
    moved = lambda new: sum(float(jnp.abs(a - b).sum()) for a, b in zip(jax.tree.leaves(new.actor), jax.tree.leaves(agent.actor)))
    np.testing.assert_allclose(moved(slow), moved(full) / 10, rtol=1e-3)
    for a, b in zip(jax.tree.leaves(slow.critic), jax.tree.leaves(full.critic)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))  # the critics move as before
    assert float(slow.log_alpha) == float(full.log_alpha)  # and the temperature


def test_the_temperature_follows_a_target_entropy_passed_in():
    cfg = small(target_entropy=-10.0)
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    batch, key = _one_batch()
    low, _ = tqc.update(agent, cfg, batch, key, target_entropy=jnp.asarray(-50.0))
    high, _ = tqc.update(agent, cfg, batch, key, target_entropy=jnp.asarray(50.0))
    assert float(low.log_alpha) < float(agent.log_alpha) < float(high.log_alpha)  # more random than asked: less bonus


def test_the_disagreement_penalty_steers_the_actor_to_where_the_critics_agree():
    batch, key = _one_batch()
    s0 = tqc.init(jax.random.PRNGKey(3), small())
    enc = small().encoder

    def spread(s):  # the critics' disagreement at the actor's mean action
        q = jnp.mean(nets.ensemble(s.critic, enc, batch.obs, nets.actor_mean(s.actor, enc, batch.obs)), axis=2)
        return float(jnp.mean(jnp.std(q, axis=0)))

    def run(cfg):
        s = s0._replace(critic=tqc.init(jax.random.PRNGKey(3), small()).critic)
        for i in range(300):
            s, m = tqc.update(s, cfg, batch, jax.random.fold_in(key, i), actor_on=1.0)
            s = s._replace(critic=s0.critic, target=s0.target)  # the same fixed critics in both runs: only the actor learns
        return spread(s), float(m["spread"])

    free, held = run(small(lr=3e-3)), run(small(lr=3e-3, disagree=100.0))
    assert held[0] < 0.95 * free[0]  # random toy critics' spread hardly depends on the action (2026-11-23 probe: -11%)
    assert free[1] > 0


def test_a_shared_encoder_reads_the_inputs_once_for_every_critic():
    lidar = EncoderSpec(lidar_beams=4, lidar_channels=1, conv=((2, 3, 1),), other_hidden=(4,))  # 4 beams, 2 other inputs
    cfg = small(shared_encoder=True, encoder=lidar)
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    assert nets.is_shared(agent.critic) and "enc" not in agent.critic
    each = tqc.init(jax.random.PRNGKey(0), small(encoder=lidar)).critic
    one = nets.share_encoder(each, which=1)
    obs, act = jax.random.normal(jax.random.PRNGKey(1), (8, 6)), jnp.zeros((8, 2))
    # critic 1 computes what it did with its own encoder; the others now read critic 1's features
    np.testing.assert_allclose(np.asarray(nets.ensemble(one, cfg.encoder, obs, act)[1]), np.asarray(nets.ensemble(each, cfg.encoder, obs, act)[1]), rtol=1e-6)
    k = jax.random.split(jax.random.PRNGKey(2), 4)
    batch = tqc.Batch(jax.random.normal(k[0], (8, 6)), jax.random.uniform(k[1], (8, 2), minval=-1, maxval=1), jnp.ones(8), jnp.full(8, 0.99), jax.random.normal(k[2], (8, 6)))
    new, m = jax.jit(lambda a, b, key: tqc.update(a, cfg, b, key))(agent, batch, k[3])
    assert all(np.isfinite(float(v)) for v in m.values()) and nets.is_shared(new.critic) and nets.is_shared(new.target)
    assert float(jnp.max(jnp.abs(new.critic["shared"]["other"][0]["w"] - agent.critic["shared"]["other"][0]["w"]))) > 0  # the encoder learns
