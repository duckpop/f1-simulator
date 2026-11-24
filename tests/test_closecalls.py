import jax.numpy as jnp
import numpy as np

from f1sim.drivers import closecalls as cc


def test_level():
    """1 - ttc / 0.4 s, clipped; nothing unless the gap closes faster than 1 m/s."""
    lv = lambda gap, closing: float(cc.level(jnp.asarray(gap), jnp.asarray(closing)))
    assert lv(10.0, 50.0) == 0.5  # 0.2 s to contact
    assert lv(10.0, 20.0) == 0.0  # 0.5 s
    assert lv(0.0, 5.0) == 1.0
    assert lv(0.1, 0.5) == 0.0  # barely closing
    assert lv(5.0, -10.0) == 0.0  # opening


def test_car_close_calls(bank):
    """Two cars nose to tail on the grid: none while still; the car behind closing at 30 m/s gets one, and so does the
    car ahead (either can make room)."""
    import jax
    from f1env.race import start_race

    race = start_race(bank, jnp.zeros(1, jnp.int32), jnp.array([5]), jax.random.PRNGKey(0), cars=2, grid=jnp.array([[0, 2]]))
    assert np.asarray(cc.cars(race)).max() == 0.0
    c = race.cars
    gap = float(np.hypot(c.x[0, 0] - c.x[0, 1], c.y[0, 0] - c.y[0, 1])) - 5.6
    moving = race._replace(cars=c._replace(vx=c.vx.at[0, 1].set(gap / 0.2)))  # 0.2 s from the car ahead
    lv = np.asarray(cc.cars(moving))[0]
    assert 0.3 < lv[1] < 0.7 and np.isclose(lv[0], lv[1])


def test_wall_close_calls():
    gap, prev = jnp.array([[1.0, 1.0, 5.0]]), jnp.array([[2.0, 1.0, 5.5]])
    racing = jnp.array([[True, True, False]])
    lv = np.asarray(cc.walls(gap, prev, racing))[0]
    assert np.isclose(lv[0], 1 - (1.0 / 20.0) / 0.4) and lv[1] == 0.0 and lv[2] == 0.0
