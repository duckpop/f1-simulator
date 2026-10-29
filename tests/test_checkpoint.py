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
