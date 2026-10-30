from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1env.race import OUT, RACING, WRECK
from f1sim import replay as rp
from f1sim.drivers import env as de


@pytest.fixture(scope="module")
def started(bank):
    cfg = de.EnvConfig(races=3, cars=1)
    st, obs = jax.jit(partial(de.reset, cfg=cfg))(bank, jax.random.PRNGKey(0))
    return cfg, st, obs


def run(bank, cfg, st, actions, key=0):
    """Steps with fixed actions (steps, R, C, 5): (state, (obs, valid, end, races restarted))."""

    @jax.jit
    def go(st, actions, key):
        def body(st, x):
            a, k = x
            st, obs, out, info = de.step(bank, st, a, k, cfg)
            return st, (obs, out.valid, out.end, info["races"], st.race.status)

        return jax.lax.scan(body, st, (actions, jax.random.split(key, actions.shape[0])))

    return go(st, actions, jax.random.PRNGKey(key))


def test_reset(started):
    cfg, st, obs = started
    assert obs.shape == (3, 1, de.OBS) and de.OBS == 209
    assert np.isfinite(np.asarray(obs)).all()
    assert all(not getattr(a, "weak_type", False) for a in jax.tree.leaves(st))
    assert (np.asarray(st.race.orders.ers) == 2).all()  # ERS deploy
    assert (np.asarray(st.race.target_laps) >= 5).all()


def test_race_lengths(bank):
    tid = jnp.zeros(4000, jnp.int32)
    length = float(bank.length[0])
    full, need = int(np.ceil(305_000 / length)), int(np.ceil(90_000 / length))
    for share in (0.0, 0.5, 1.0):
        laps = np.asarray(de.race_laps(bank, tid, jax.random.PRNGKey(1), de.EnvConfig(pit_share=share)))
        assert laps.min() >= 5 and laps.max() <= full
        long = (laps > need).mean()
        assert abs(long - share) < 0.03
    assert np.unique(laps).size > 10


def test_lights_then_driving(bank, started):
    cfg, st, _ = started
    a = jnp.zeros((300, 3, 1, 5)).at[..., 1].set(0.5)  # half throttle, straight on
    st2, (obs, valid, end, races, status) = run(bank, cfg, st, a)
    valid = np.asarray(valid)[:, :, 0]
    assert not valid[0].any()  # on the grid, lights on
    first = valid.argmax(axis=0)
    assert (valid[-1]).all() and (first > 80).all()  # five lights, then the hold
    assert all(valid[f:, r].all() for r, f in enumerate(first))
    assert (np.asarray(st2.race.line) > np.asarray(st.race.line)).all()
    assert np.isfinite(np.asarray(obs)).all()


def test_standing_still_costs_from_lights_out(bank, started):
    """A car that stays on the grid after the lights go out pays the standstill term every step it's in control; one
    that drives off pays it only for the first moment."""
    from f1sim.drivers import rewards as rw

    cfg, st, _ = started
    k = rw.TERMS.index("standstill")

    @jax.jit
    def stood(st, throttle):
        def body(st, kk):
            a = jnp.zeros((3, 1, 5)).at[..., 1].set(throttle)
            st, _, out, _ = de.step(bank, st, a, kk, cfg)
            return st, (out.amounts[..., k], out.valid)

        return jax.lax.scan(body, st, jax.random.split(jax.random.PRNGKey(0), 300))[1]

    still, valid = (np.asarray(x)[:, :, 0] for x in stood(st, -1.0))  # throttle off
    paid = np.isclose(still, 0.05)
    assert valid.any(axis=0).all() and still[~valid].sum() == 0
    assert all(valid[:, r].sum() - paid[:, r].sum() <= 1 for r in range(3))  # every step but the stuck-out (a crash)
    going, valid = (np.asarray(x)[:, :, 0] for x in stood(st, 1.0))
    assert going.sum(axis=0).max() <= 0.3  # flat out: only the first moment off the line


def test_crash_ends_the_episode_and_the_car_rejoins(bank, started):
    cfg, st, _ = started
    a = jnp.zeros((1200, 3, 1, 5)).at[..., 0].set(1.0).at[..., 1].set(1.0)  # full lock, full throttle
    _, (_, valid, end, races, status) = run(bank, cfg, st, a)
    valid, end, status = np.asarray(valid)[:, :, 0], np.asarray(end)[:, :, 0], np.asarray(status)[:, :, 0]
    assert int(np.asarray(races).sum()) == 0  # one car, but the race goes on while it rejoins
    for r in range(3):
        t = np.flatnonzero(end[:, r] == rp.TERMINAL)
        assert t.size, "the car should crash"
        t = t[0]
        assert status[t, r] in (WRECK, OUT) and not valid[t + 1, r]
        back = t + 1 + np.flatnonzero(valid[t + 1 :, r])
        assert back.size and 95 <= back[0] - t <= 110  # ~5 s later, from the pit exit
        assert status[back[0], r] == RACING


def test_time_limit_cuts_and_restarts_the_race(bank, started):
    cfg, st, _ = started
    a = jnp.zeros((200, 3, 1, 5)).at[..., 1].set(0.3)
    st2, _ = run(bank, cfg, st, a)
    cut = st2._replace(limit=st2.limit.at[1].set(st2.race.clock[1] + 0.01))  # race 1 reaches its limit this step
    st3, (_, valid, end, races, _) = run(bank, cfg, cut, a[:1])
    assert int(np.asarray(end)[0, 1, 0]) == rp.CUT and int(np.asarray(races).sum()) == 1
    assert float(st3.race.clock[1]) < 0 and int(st3.tick[1]) == 0  # a new race on the grid
    assert float(st3.race.clock[0]) > 0 and int(st3.tick[0]) == int(st2.tick[0]) + 1  # the others carry on
    np.testing.assert_array_equal(np.asarray(end)[0, [0, 2], 0], rp.CONTINUE)


def test_qualifying_sessions(bank):
    from f1env.race import QUALIFYING

    cfg = de.EnvConfig(races=2, cars=4, quali_share=1.0, quali_s=300.0)
    st, _ = jax.jit(partial(de.reset, cfg=cfg))(bank, jax.random.PRNGKey(3))
    assert (np.asarray(st.race.session) == QUALIFYING).all() and np.isfinite(np.asarray(st.plan)).all()
    a = jnp.zeros((800, 2, 4, 5)).at[..., 1].set(0.4)
    st2, (_, valid, end, races, _) = run(bank, cfg, st, a)
    valid = np.asarray(valid)
    assert not valid[0].any()  # every car in its garage
    out = valid.argmax(axis=0)  # leaving the garage a few seconds apart, in a random order
    assert valid[-1].any() and len(np.unique(out[valid.any(axis=0)])) > 1
    assert (np.asarray(st2.race.pits.stops) == 0).all() and int(np.asarray(races).sum()) == 0


def test_mixed_sessions(bank):
    from f1env.race import QUALIFYING

    cfg = de.EnvConfig(races=200, cars=2, quali_share=0.2)
    st, _ = jax.jit(partial(de.reset, cfg=cfg))(bank, jax.random.PRNGKey(4))
    is_q = np.asarray(st.race.session) == QUALIFYING
    assert 0.1 < is_q.mean() < 0.3
    assert (np.asarray(st.race.clock)[is_q] == 0).all() and (np.asarray(st.race.clock)[~is_q] < 0).all()  # garages / lights
    assert np.isinf(np.asarray(st.plan)[~is_q]).all() and np.isfinite(np.asarray(st.plan)[is_q]).all()
