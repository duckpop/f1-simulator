import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1env.sensors import make_bank
from f1sim import replay as rp
from f1sim.drivers import env as de
from f1sim.drivers.tracks import Rotation, bank_sizes, pad_bank, rotate_races


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    from f1env.tracks.pool import build_pool

    out = tmp_path_factory.mktemp("pool")
    build_pool(out, n_train=5, n_eval=0, wide=(), log=None)
    return out


def run(bank, steps=150):
    cfg = de.EnvConfig(races=2, cars=2)
    st, _ = de.compiled("reset", cfg)(bank, jax.random.PRNGKey(0), tid=jnp.array([0, 1]))

    @jax.jit
    def go(bank, st):
        def body(st, k):
            a = jnp.zeros((2, 2, 5)).at[..., 1].set(0.5).at[..., 0].set(0.1)
            return de.step(bank, st, a, k, cfg)[0], None

        return jax.lax.scan(body, st, jax.random.split(jax.random.PRNGKey(1), steps))[0]

    return go(bank, st).race


def test_padding_changes_nothing(tracks):
    bank = make_bank(tracks)
    cells, line, lanes = bank_sizes(bank)
    padded = pad_bank(bank, (cells + 5000, line + 300, lanes + 20))
    assert bank_sizes(padded) == (cells + 5000, line + 300, lanes + 20)
    a, b = run(bank), run(padded)
    for name in ("line", "s", "status", "battery"):
        np.testing.assert_array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name)), err_msg=name)
    np.testing.assert_array_equal(np.asarray(a.cars.x), np.asarray(b.cars.x))


def test_rotation_swaps_the_older_set(pool):
    rot = Rotation(pool, "normal", n=2, seed=0, log=None)
    assert rot.rotates and rot.bank.length.shape == (4,) and list(np.asarray(rot.pick)) == [0, 2]
    shapes = jax.tree.map(lambda a: a.shape, rot.bank)
    _, replaced = rot.rotate()  # the second set was loaded but never raced on: nothing to replace
    assert replaced.size == 0 and list(np.asarray(rot.pick)) == [2, 2]
    bank, replaced = rot.rotate()  # now the first set (raced on one rotation ago) is replaced
    assert list(replaced) == [0, 1] and list(np.asarray(rot.pick)) == [0, 2]
    bank, replaced = rot.rotate()
    assert list(replaced) == [2, 3]
    assert len(rot.seen) == 5  # every track came round before any came back
    assert jax.tree.map(lambda a: a.shape, bank) == shapes or bank_sizes(bank) == rot.sizes  # same shapes: no recompile


def test_no_rotation_without_a_pool_or_with_a_small_one(pool):
    assert not Rotation(pool, "normal", n=8, log=None).rotates  # 5 tracks, 8 wanted: one fixed set


def test_rotate_races_restarts_only_races_on_replaced_tracks(pool):
    rot = Rotation(pool, "normal", n=2, seed=1, log=None)
    cfg = de.EnvConfig(races=4, cars=1)
    st, obs = de.compiled("reset", cfg)(rot.bank, jax.random.PRNGKey(0), tid=jnp.array([0, 1, 0, 1]), pick=rot.pick)
    a = jnp.zeros((300, 4, 1, 5)).at[..., 1].set(0.4)

    @jax.jit
    def go(bank, st, obs):
        def body(c, x):
            st, obs = c
            st, obs, _, _ = de.step(bank, st, x, jax.random.PRNGKey(2), cfg)
            return (st, obs), None

        return jax.lax.scan(body, (st, obs), a)[0]

    st, obs = go(rot.bank, st, obs)
    rot.rotate()  # to the fresh second set: no race is on a replaced track yet
    st = st._replace(race=st.race._replace(tid=jnp.array([0, 1, 2, 3])))  # pretend two races already moved on
    bank, st2, obs2, hit = rotate_races(rot, st, obs, jax.random.PRNGKey(3), cfg)
    assert list(hit) == [True, True, False, False]  # the first set was just replaced
    tid = np.asarray(st2.race.tid)
    assert list(np.asarray(rot.pick)) == [0, 2]  # the fresh set went into slots 0-1 and is the active one
    assert set(tid[:2]) <= {0, 1} and list(tid[2:]) == [2, 3]  # restarted on the fresh set; the others untouched
    assert (np.asarray(st2.race.clock)[:2] < 0).all() and (np.asarray(st2.race.clock)[2:] > 0).all()
    np.testing.assert_array_equal(np.asarray(obs2)[2:], np.asarray(obs)[2:])


def test_cut_last():
    buf = rp.new(8, 3, 2, 1)
    end = jnp.array([rp.CONTINUE, rp.TERMINAL, rp.CONTINUE], jnp.int8)
    buf = rp.add(buf, jnp.zeros((3, 2)), jnp.zeros((3, 1)), jnp.zeros(3), jnp.array([True, True, False]), end)
    buf = rp.cut_last(buf, jnp.array([True, True, True]))
    assert list(np.asarray(buf.end[0])) == [rp.CUT, rp.TERMINAL, rp.CONTINUE]  # a terminal stays; an invalid row too
