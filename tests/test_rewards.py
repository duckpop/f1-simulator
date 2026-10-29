from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from f1sim.drivers import rewards as rw

TERM = {t: i for i, t in enumerate(rw.TERMS)}


def events(gained=0, lost=0, place=5):
    z = jnp.zeros((1, 1))
    i = lambda x: jnp.full((1, 1), x, jnp.int32)
    rec = SimpleNamespace(gained=i(gained), lost=i(lost), place=i(place), under_attack=z > 0, inside_line=z > 0)
    dyn = SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z)
    return SimpleNamespace(
        records=rec, dynamics=dyn, progress=z, crashed=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, respawned=z > 0
    )


def run(steps, start_place=5):
    """steps: (gained, lost, place) per step; the total amount of each term."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), start_place))
    total = np.zeros(len(rw.TERMS))
    for g, l, p in steps:
        ev = events(g, l, p)
        out, mem = rw.amounts(ev, a, a, mem, jnp.ones((1, 1), bool), ev.respawned)
        total += np.asarray(out[0, 0])
    return total


def test_overtake_paid_once_held_for_two_seconds():
    hold = [(1, 0, 4)] + [(0, 0, 4)] * 39
    assert run(hold)[TERM["overtake"]] == 0  # 1.95 s
    assert run(hold + [(0, 0, 4)])[TERM["overtake"]] == 1


def test_place_given_back_at_once_cancels_the_overtake():
    t = run([(1, 0, 4)] + [(0, 0, 4)] * 10 + [(0, 1, 5)] + [(0, 0, 5)] * 60)
    assert t[TERM["overtake"]] == 0 and t[TERM["place_lost"]] == 0


def test_swapping_places_back_and_forth_pays_nothing_extra():
    still = [(0, 0, 4)] * 50
    t = run([(1, 0, 4)] + still + [(0, 1, 5)] + [(0, 0, 5)] * 50 + [(1, 0, 4)] + still)
    assert t[TERM["overtake"]] == 1 and t[TERM["place_lost"]] == 1


def test_profile_weights_scale_style_terms_only():
    w = dict(zip(rw.TERMS, rw.weights(None, {"overtake": 2.0, "smoothness": 0.5})))
    assert w["overtake"] == 2 * rw.BASE["overtake"] and w["place_lost"] == rw.BASE["place_lost"]
    assert w["inputs"] == 0.5 * rw.BASE["inputs"] and w["slip"] == 0.5 * rw.BASE["slip"]
    assert w["progress"] == rw.BASE["progress"] and w["crash"] == -70.0
    assert np.isclose(rw.BASE["progress"] * 86.1 * 0.05, 1.0)  # a step at top speed is worth +1
