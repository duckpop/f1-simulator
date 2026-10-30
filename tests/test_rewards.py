from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from f1sim.drivers import rewards as rw

TERM = {t: i for i, t in enumerate(rw.TERMS)}


def events(gained=0, lost=0, place=5, progress=0.0, off=False):
    z = jnp.zeros((1, 1))
    i = lambda x: jnp.full((1, 1), x, jnp.int32)
    rec = SimpleNamespace(gained=i(gained), lost=i(lost), place=i(place), under_attack=z > 0, inside_line=z > 0)
    dyn = SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z)
    return SimpleNamespace(
        records=rec, dynamics=dyn, progress=z + progress, crashed=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, respawned=z > 0,
        off_track=jnp.full((1, 1), off),
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


def test_qualifying_standing_and_laps():
    """Points 21 for P1 down to 0 for P22 (no time: 0), +3 into the top 16, +5 into the top 10, +3 per personal best,
    each paid when the standing changes; over a session a car collects what its final place is worth."""
    inf = np.inf
    a = SimpleNamespace(steer=jnp.zeros((1, 22)), throttle=jnp.zeros((1, 22)), brake=jnp.zeros((1, 22)))
    z = jnp.zeros((1, 22))
    i = lambda x: jnp.full((1, 22), x, jnp.int32)
    ev = SimpleNamespace(
        records=SimpleNamespace(gained=i(0), lost=i(0), place=i(1), under_attack=z > 0, inside_line=z > 0),
        dynamics=SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z),
        progress=z, crashed=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, off_track=z > 0,
    )
    mem = rw.new_memory(jnp.ones((1, 22)))
    on = jnp.ones(1, bool)
    step = lambda old, new, q=on: np.asarray(rw.amounts(ev, a, a, mem, jnp.ones((1, 22), bool), z > 0, quali=(q, jnp.asarray([old]), jnp.asarray([new])))[0][0])
    times = [80.0 + k for k in range(22)]  # car k is P(k + 1)
    pts, top16, top10 = (np.asarray(x[0]) for x in rw.quali_standing(jnp.asarray([times])))
    assert pts[0] == 21 and pts[21] == 0 and top16.sum() == 16 and top10.sum() == 10
    # car 21 (last) sets 79.0: P1. It gains 21 points, both groups and a lap; everyone else drops a place
    new = list(times)
    new[21] = 79.0
    out = step(times, new)
    T = lambda t: out[:, TERM[t]]
    assert T("quali_place")[21] == 21 and T("quali_q2")[21] == 1 and T("quali_q3")[21] == 1 and T("quali_lap")[21] == 1
    assert all(T("quali_place")[:21] == -1) and T("quali_q3")[9] == -1 and T("quali_q2")[15] == -1  # P10 -> P11, P16 -> P17
    assert T("quali_lap")[:21].sum() == 0
    # a first time counts as a personal best; a slower lap doesn't
    first = [inf] * 22
    first[3] = 90.0
    assert step([inf] * 22, first)[3, TERM["quali_lap"]] == 1 and step([inf] * 22, first)[3, TERM["quali_place"]] == 21
    assert step(times, list(times))[:, TERM["quali_lap"]].sum() == 0
    # not in a race
    assert np.abs(step(times, new, jnp.zeros(1, bool))[:, [TERM[t] for t in ("quali_place", "quali_q2", "quali_q3", "quali_lap")]]).sum() == 0
    w = dict(zip(rw.TERMS, rw.weights()))
    assert (w["quali_place"], w["quali_q2"], w["quali_q3"], w["quali_lap"]) == (1.0, 3.0, 5.0, 3.0)
    # standing still: per second, only for the cars flagged
    stopped = jnp.zeros((1, 22), bool).at[0, 3].set(True)
    out = np.asarray(rw.amounts(ev, a, a, mem, jnp.ones((1, 22), bool), z > 0, stopped=stopped)[0][0])
    assert out[3, TERM["standstill"]] == 0.05 and out[:, TERM["standstill"]].sum() == 0.05 and w["standstill"] == -20.0


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


def test_no_progress_off_the_track():
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    on, _ = rw.amounts(events(progress=4.0), a, a, mem, jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool))
    off, _ = rw.amounts(events(progress=4.0, off=True), a, a, mem, jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool))
    assert float(on[0, 0, TERM["progress"]]) == 4.0 and float(on[0, 0, TERM["off_track"]]) == 0.0
    assert float(off[0, 0, TERM["progress"]]) == 0.0 and np.isclose(float(off[0, 0, TERM["off_track"]]), 0.05)
    w = dict(zip(rw.TERMS, rw.weights()))
    assert w["off_track"] == -2.0  # common to every profile
    assert all(dict(zip(rw.TERMS, rw.weights(None, {s: 2.0 for s in rw.STYLES})))["off_track"] == -2.0 for _ in [0])
