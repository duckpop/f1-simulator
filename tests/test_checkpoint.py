import jax
import numpy as np
import pytest

from f1sim import checkpoint, tqc


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
