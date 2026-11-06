from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from f1sim.drivers import rewards as rw

TERM = {t: i for i, t in enumerate(rw.TERMS)}


def events(gained=0, lost=0, place=5, progress=0.0, off=False):
    z = jnp.zeros((1, 1))
    i = lambda x: jnp.full((1, 1), x, jnp.int32)
    rec = SimpleNamespace(gained=i(gained), lost=i(lost), place=i(place), under_attack=z > 0, inside_line=z > 0)
    dyn = SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z, slide_front=z, slide_rear=z)
    return SimpleNamespace(
        records=rec, dynamics=dyn, progress=z + progress, crashed=z > 0, crash_blame=z, black_flag=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, respawned=z > 0,
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
        dynamics=SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z, slide_front=z, slide_rear=z),
        progress=z, crashed=z > 0, crash_blame=z, black_flag=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, off_track=z > 0,
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
    assert w["progress"] == rw.BASE["progress"] and w["crash"] == -2500.0
    assert rw.BASE["progress"] == 1.0  # a step at reference pace is worth +1


def test_no_progress_off_the_track():
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    on, _ = rw.amounts(events(progress=4.0), a, a, mem, jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool))
    off, _ = rw.amounts(events(progress=4.0, off=True), a, a, mem, jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool))
    assert np.isclose(float(on[0, 0, TERM["progress"]]), 4.0 / (86.1 * 0.05)) and float(on[0, 0, TERM["off_track"]]) == 0.0  # no pace: top speed
    assert float(off[0, 0, TERM["progress"]]) == 0.0 and np.isclose(float(off[0, 0, TERM["off_track"]]), 0.05)
    w = dict(zip(rw.TERMS, rw.weights()))
    assert w["off_track"] == -2.0  # common to every profile
    assert all(dict(zip(rw.TERMS, rw.weights(None, {s: 2.0 for s in rw.STYLES})))["off_track"] == -2.0 for _ in [0])


def test_progress_against_the_reference_pace():
    """A step at the reference pace (the lap simulation's speed there, or the limit) pays +1, wherever it is."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    on, f = jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool)
    for v in (20.0, 43.0, 80.0):  # a hairpin, behind the safety car, a straight
        out, _ = rw.amounts(events(progress=v * 0.05), a, a, mem, on, f, pace=jnp.full((1, 1), v))
        assert np.isclose(float(out[0, 0, TERM["progress"]]), 1.0)
    slow, _ = rw.amounts(events(progress=0.5 * 20.0 * 0.05), a, a, mem, on, f, pace=jnp.full((1, 1), 20.0))
    assert np.isclose(float(slow[0, 0, TERM["progress"]]), 0.5)


def test_tyre_wear_beyond_the_rated_wear():
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    on, f = jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool)
    ev = events()
    ev.tyre_wear = jnp.full((1, 1), 3e-5)
    full, _ = rw.amounts(ev, a, a, mem, on, f)
    beyond, _ = rw.amounts(ev, a, a, mem, on, f, rated_wear=jnp.full((1, 1), 1e-5))
    under, _ = rw.amounts(ev, a, a, mem, on, f, rated_wear=jnp.full((1, 1), 5e-5))
    assert np.isclose(float(full[0, 0, TERM["tyre_wear"]]), 3e-3) and np.isclose(float(beyond[0, 0, TERM["tyre_wear"]]), 2e-3)
    assert float(under[0, 0, TERM["tyre_wear"]]) == 0.0  # gentler than rated: no cost (and no credit)


def test_every_car_in_a_crash_pays_the_full_cost():
    """The car that drove into the other and the car it hit both pay the full crash cost, as does a crash alone."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    on, f = jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool)
    for blame in (1.0, 0.0, 0.5):
        ev = events()
        ev.crashed, ev.crash_blame = jnp.ones((1, 1), bool), jnp.full((1, 1), blame)
        out, _ = rw.amounts(ev, a, a, mem, on, f)
        assert np.isclose(float(out[0, 0, TERM["crash"]]), 1.0)


def test_a_black_flag_costs_as_much_as_a_crash():
    """The 4th warning's black flag is its own term, at the crash's base value: never a cheaper way out."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    ev = events()
    ev.black_flag = jnp.ones((1, 1), bool)
    out, _ = rw.amounts(ev, a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool))
    assert float(out[0, 0, TERM["black_flag"]]) == 1.0
    w = rw.weights()
    assert w[TERM["black_flag"]] == w[TERM["crash"]] == -2500.0


def test_the_reward_scale_multiplies_every_weight():
    w, w10 = rw.weights(), rw.weights(scale=0.1)
    assert np.allclose(w10, 0.1 * w)


def test_a_sector_for_each_third_of_a_lap_driven():
    """+1 sector each time the car's own progress on the track this episode passes another third of a lap; going back
    and forth doesn't pay twice, off-track progress doesn't count, a new episode starts again."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    on, f, third = jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), jnp.full((1, 1), 1000.0)
    paid = []
    for metres, off in [(600.0, False), (600.0, False), (-300.0, False), (300.0, False), (900.0, True), (900.0, False)]:
        out, mem = rw.amounts(events(progress=metres, off=off), a, a, mem, on, f, sector_len=third)
        paid.append(int(out[0, 0, TERM["sector"]]))
    assert paid == [0, 1, 0, 0, 0, 1]  # 600, 1200 (a third), 900, 1200 (already paid), off track, 2100 (the second)
    out, mem = rw.amounts(events(progress=500.0), a, a, mem, on, jnp.ones((1, 1), bool), sector_len=third)
    assert int(out[0, 0, TERM["sector"]]) == 0 and float(mem.driven[0, 0]) == 0.0  # respawned: a new episode


def test_too_close_behind_under_the_safety_car():
    """With the safety car out, under 0.5 s behind the car ahead costs per second; further back, or without it, not."""
    from f1env import racecontrol as rc

    speed = jnp.full((1, 3), 40.0)
    for phase, want in ((rc.OUT, [True, False, False]), (rc.NONE, [False, False, False])):
        race = SimpleNamespace(cars=SimpleNamespace(vx=speed, vy=jnp.zeros((1, 3))), sc=SimpleNamespace(phase=jnp.array([phase])))
        assert list(np.asarray(rw.too_close(race, jnp.array([[10.0, 30.0, jnp.inf]]))[0])) == want  # 0.25 s, 0.75 s, nobody
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    out, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), close=jnp.ones((1, 1), bool))
    assert np.isclose(float(out[0, 0, TERM["sc_close"]]), 0.05)


def test_warnings_cost_a_growing_share_of_a_black_flag():
    """The 1st, 2nd and 3rd warnings cost 25%, 50% and 75% of a black flag; the 4th is the black flag itself, so a
    disqualification costs 250% of it in all."""
    i = lambda x: jnp.full((1, 1), x, jnp.int32)
    share = lambda bits, before: float(rw.warning_shares(i(bits), i(before))[0, 0])
    assert [share(1, k) for k in range(4)] == [0.25, 0.5, 0.75, 0.0]
    assert share(1 | 4, 1) == 1.25  # two at once: the 2nd and the 3rd
    assert share(0, 2) == 0.0
    w = rw.weights()
    assert w[TERM["warning"]] == w[TERM["black_flag"]]
    total = sum(share(1, k) for k in range(4)) * w[TERM["warning"]] + w[TERM["black_flag"]]
    assert total == 2.5 * w[TERM["black_flag"]]

    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    ev = events()
    ev.warned = i(1)
    out, _ = rw.amounts(ev, a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), warned_before=i(1))
    assert float(out[0, 0, TERM["warning"]]) == 0.5
