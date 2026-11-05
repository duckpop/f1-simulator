import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1sim import checkpoint, nets, tqc
from f1sim.drivers.config import load


def test_round_trip(tmp_path):
    cfg = tqc.TQCConfig(obs_dim=4, act_dim=2, critics=2, quantiles=3, actor_hidden=(8,), critic_hidden=(8,))
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    checkpoint.save(tmp_path / "a.npz", agent, {"profile": "middle", "env_steps": 5})
    back, meta = checkpoint.load(tmp_path / "a.npz", tqc.init(jax.random.PRNGKey(1), cfg))
    assert meta == {"profile": "middle", "env_steps": 5} == checkpoint.meta(tmp_path / "a.npz")
    for a, b in zip(jax.tree.leaves(agent), jax.tree.leaves(back)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        checkpoint.load(tmp_path / "a.npz", tqc.init(jax.random.PRNGKey(1), cfg._replace(actor_hidden=(16,))))


def test_best_bar_only_from_the_same_kind_of_eval(tmp_path):
    cfg = tqc.TQCConfig(obs_dim=4, act_dim=2, critics=2, quantiles=3, actor_hidden=(8,), critic_hidden=(8,))
    agent = tqc.init(jax.random.PRNGKey(0), cfg)
    assert checkpoint.best_bar(tmp_path / "none.npz", stage="A") is None
    checkpoint.save(tmp_path / "best.npz", agent, {"stage": "A", "eval_kind": "alone", "eval_laps": 5, "eval": {"finish_rate": 0.8}})
    assert checkpoint.best_bar(tmp_path / "best.npz", stage="A", eval_kind="alone", eval_laps=5) == {"finish_rate": 0.8}
    assert checkpoint.best_bar(tmp_path / "best.npz", stage="A", eval_kind="alone", eval_laps=50) is None  # 5-lap evals aren't a bar for 50


def test_a_shallower_checkpoint_loads_deepened(tmp_path):
    """Saved with fewer hidden layers, loaded with more: the extra layers are identities, so the driver and the critics
    give exactly the same outputs as before."""
    cfg = load("middle").tqc._replace(actor_hidden=(32, 32), critic_hidden=(48, 48))
    small = tqc.init(jax.random.PRNGKey(3), cfg)
    checkpoint.save(tmp_path / "s.npz", small, {"profile": "middle"})
    deep_cfg = cfg._replace(actor_hidden=(32, 32, 32), critic_hidden=(48, 48, 48))
    deep, meta = checkpoint.load(tmp_path / "s.npz", tqc.init(jax.random.PRNGKey(1), deep_cfg), deep_cfg)
    assert meta["deepened"] == [1, 1] and len(deep.actor["head"]) == len(small.actor["head"]) + 1
    obs = jax.random.normal(jax.random.PRNGKey(5), (7, cfg.obs_dim))
    act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(6), (7, cfg.act_dim)))
    assert jnp.allclose(nets.actor_mean(small.actor, cfg.encoder, obs), nets.actor_mean(deep.actor, deep_cfg.encoder, obs), atol=1e-5)
    assert jnp.allclose(nets.ensemble(small.critic, cfg.encoder, obs, act), nets.ensemble(deep.critic, deep_cfg.encoder, obs, act), atol=1e-4)


def test_an_older_checkpoint_loads_with_new_inputs(tmp_path):
    """Saved with 25 fewer inputs (and a layer fewer), loaded with them: they get zero weights, so whatever they are the
    driver and the critics give what they did on the old inputs."""
    full = load("middle").tqc._replace(actor_hidden=(32, 32), critic_hidden=(48, 48))
    old_cfg = full._replace(obs_dim=full.obs_dim - 25, actor_hidden=(32,), critic_hidden=(48,))
    small = tqc.init(jax.random.PRNGKey(3), old_cfg)
    checkpoint.save(tmp_path / "s.npz", small, {"profile": "middle"})
    big, meta = checkpoint.load(tmp_path / "s.npz", tqc.init(jax.random.PRNGKey(1), full), full)
    assert meta["widened"] == 25 and meta["deepened"] == [1, 1]
    obs = jax.random.normal(jax.random.PRNGKey(5), (7, full.obs_dim))
    act = jnp.tanh(jax.random.normal(jax.random.PRNGKey(6), (7, full.act_dim)))
    old_obs = obs[:, : old_cfg.obs_dim]
    assert jnp.allclose(nets.actor_mean(small.actor, old_cfg.encoder, old_obs), nets.actor_mean(big.actor, full.encoder, obs), atol=1e-5)
    assert jnp.allclose(nets.ensemble(small.critic, old_cfg.encoder, old_obs, act), nets.ensemble(big.critic, full.encoder, obs, act), atol=1e-4)
