import jax
import jax.numpy as jnp
import numpy as np

from f1env import orders as od
from f1env import tyres as ty
from f1env.race import RaceConfig, start_race
from f1sim.drivers import principal


def race(bank, laps=20):
    st = start_race(bank, jnp.array([0, 1]), jnp.array([laps, laps]), jax.random.PRNGKey(0), RaceConfig(), 1)
    return st._replace(weather=st.weather._replace(raining=jnp.zeros(2, bool), wetness=jnp.zeros(2), change_lap=jnp.full(2, 999)))


def calls(bank, st, worn):
    return np.asarray(principal.calls(bank, st, jnp.full((2, 1), worn)))[:, 0]


def test_boxes_worn_tyres_for_the_next_harder_compound(bank):
    st = race(bank)
    assert (calls(bank, st, 0.10) == -1).all()  # new mediums
    worn = st._replace(tyres=st.tyres._replace(life=jnp.full((2, 1), 0.05)))
    assert (calls(bank, worn, 0.10) == 1 + ty.HARD).all()
    soft = worn._replace(tyres=worn.tyres._replace(compound=jnp.full((2, 1), ty.SOFT)))
    assert (calls(bank, soft, 0.10) == 1 + ty.MEDIUM).all()


def test_threshold_is_per_stint(bank):
    st = race(bank)
    st = st._replace(tyres=st.tyres._replace(life=jnp.full((2, 1), 0.5)))
    assert (calls(bank, st, 0.10) == -1).all() and (calls(bank, st, 0.9) == 1 + ty.HARD).all()
    w = principal.draw_worn(jax.random.PRNGKey(0), (1000,))
    assert 0.08 <= float(w.min()) and float(w.max()) <= 0.12


def test_wets_for_rain(bank):
    st = race(bank)
    wet = st._replace(weather=st.weather._replace(raining=jnp.ones(2, bool), wetness=jnp.ones(2)))
    assert (calls(bank, wet, 0.10) == 1 + ty.WET).all()


def test_orders_fixed(bank):
    o = principal.fixed_orders(race(bank)).orders
    assert (np.asarray(o.ers) == od.DEPLOY).all() and (np.asarray(o.pace) == od.NORMAL).all() and (np.asarray(o.team) == od.NO_ORDER).all()
