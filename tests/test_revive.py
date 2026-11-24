import jax
import jax.numpy as jnp
import numpy as np

from f1sim import nets, revive

SPEC = nets.EncoderSpec()  # no lidar branch: the inputs go straight to the head


def _kill(net, layer, units, lead=()):
    """The units' biases far below zero: they fire on nothing."""
    head = [dict(p) for p in net["head"]]
    b = np.array(head[layer]["b"])
    b[..., units] = -1e3
    head[layer]["b"] = jnp.asarray(b)
    return {**net, "head": head}


def test_dead_actor_units_come_back_and_nothing_changes():
    key = jax.random.PRNGKey(0)
    actor = _kill(_kill(nets.actor_init(key, SPEC, 6, 2, (16, 16, 16)), 2, [0, 3, 5, 9]), 1, [2, 3])
    twin = jax.tree.map(lambda a: a * 1.01, actor)  # an averaged copy: nearly the same weights
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (512, 6)))
    before, before_twin = revive.output(actor, SPEC, obs), revive.output(twin, SPEC, obs)
    dead = [~(a | b) for a, b in zip(revive.firing(actor, SPEC, obs), revive.firing(twin, SPEC, obs))]  # the killed ones, and any a small random net has
    assert dead[2][[0, 3, 5, 9]].all() and dead[1][[2, 3]].all()
    new, new_twin, rep = revive.revive(actor, SPEC, obs, twin=twin, seed=3)
    assert not rep["refused"] and rep["revived"] == [int(d.sum()) for d in dead] and rep["units"] == [16, 16, 16]
    assert np.array_equal(revive.output(new, SPEC, obs), before) and np.array_equal(revive.output(new_twin, SPEC, obs), before_twin)
    assert all(f.sum() >= a for f, a in zip(revive.firing(new, SPEC, obs), rep["alive"]))  # none lost
    assert revive.firing(new, SPEC, obs)[2][[0, 3, 5, 9]].sum() >= 3  # they fire again (a fresh unit may be silent on a sample)
    w3, out = np.asarray(new["head"][2]["w"]), np.asarray(new["head"][3]["w"])
    assert np.all(out[[0, 3, 5, 9]] == 0) and np.all(np.asarray(new["head"][2]["b"])[[0, 3, 5, 9]] == 0)
    assert np.array_equal(w3[:, [0, 3, 5, 9]], np.asarray(new_twin["head"][2]["w"])[:, [0, 3, 5, 9]])  # the same new units in both
    keep = np.ix_(~dead[1], ~dead[2])  # from a live unit to a live unit: untouched
    assert np.array_equal(w3[keep], np.asarray(actor["head"][2]["w"])[keep])
    # a revived unit learns: its outgoing weights get a gradient, which a dead unit's don't
    loss = lambda p: jnp.sum(nets.mlp(p["head"], jnp.asarray(obs)) ** 2)
    g_new, g_old = jax.grad(loss)(new)["head"][3]["w"], jax.grad(loss)(actor)["head"][3]["w"]
    assert float(jnp.abs(g_old[jnp.array([0, 3, 5, 9])]).max()) == 0.0 and float(jnp.abs(g_new[jnp.array([0, 3, 5, 9])]).max()) > 0.0
    again, _, rep2 = revive.revive(new, SPEC, obs, twin=new_twin, seed=4)
    assert sum(rep2["revived"]) < sum(rep["revived"])  # at most a fresh unit that happens to be silent on this sample


def test_dead_critic_units_come_back_per_critic():
    critics = nets.ensemble_init(jax.random.PRNGKey(0), 3, SPEC, 6, 2, (16, 16), 5)
    head = [dict(p) for p in critics["head"]]
    b = np.array(head[1]["b"])
    b[0, [1, 2]], b[2, [7]] = -1e3, -1e3  # two dead in the first critic, one in the third
    head[1]["b"] = jnp.asarray(b)
    critics = {**critics, "head": head}
    target = jax.tree.map(jnp.copy, critics)
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (256, 6)))
    act = np.asarray(jax.random.uniform(jax.random.PRNGKey(2), (256, 2), minval=-1, maxval=1))
    before = revive.output(critics, SPEC, obs, act)
    new, new_target, rep = revive.revive(critics, SPEC, obs, act, twin=target, seed=1)
    assert rep["revived"][1] >= 3 and rep["units"] == [48, 48] and not rep["refused"]
    assert np.array_equal(revive.output(new, SPEC, obs, act), before) and np.array_equal(revive.output(new_target, SPEC, obs, act), before)
    out = np.asarray(new["head"][2]["w"])
    assert np.all(out[0, [1, 2]] == 0) and np.all(out[2, 7] == 0) and np.any(out[1] != 0) and np.any(out[0, 0] != 0)


def test_nothing_dead_changes_nothing():
    actor = nets.actor_init(jax.random.PRNGKey(0), SPEC, 6, 2, (16, 16))
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (2048, 6)) * 3)
    new, twin, rep = revive.revive(actor, SPEC, obs)
    if not any(rep["revived"]):
        assert new is actor and twin is None
    assert "alive" in revive.describe(rep)


def test_reviving_again_reuses_the_compiled_functions():
    """New weights, the same shapes: no new compile (a compile per call, weights baked in, kept memory every eval)."""
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (512, 6)))
    sizes = []
    for seed in range(3):
        actor = nets.actor_init(jax.random.PRNGKey(seed), SPEC, 6, 2, (16, 16, 16))
        revive.revive(actor, SPEC, obs, twin=actor, seed=seed)
        sizes.append((revive._out_actor._cache_size(), revive._fires_actor._cache_size()))
    assert sizes[0] == sizes[1] == sizes[2]


def test_dead_units_come_back_in_critics_sharing_an_encoder():
    critics = nets.ensemble_init(jax.random.PRNGKey(0), 3, SPEC, 6, 2, (16, 16), 5, shared=True)
    head = [dict(p) for p in critics["head"]]
    b = np.array(head[1]["b"])
    b[1, [3, 4]] = -1e3
    head[1]["b"] = jnp.asarray(b)
    critics = {**critics, "head": head}
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (256, 6)))
    act = np.asarray(jax.random.uniform(jax.random.PRNGKey(2), (256, 2), minval=-1, maxval=1))
    before = revive.output(critics, SPEC, obs, act)
    new, _, rep = revive.revive(critics, SPEC, obs, act, seed=1)
    assert rep["revived"][1] >= 2 and nets.is_shared(new)
    assert np.array_equal(revive.output(new, SPEC, obs, act), before)
