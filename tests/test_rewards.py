from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from f1sim.drivers import rewards as rw

TERM = {t: i for i, t in enumerate(rw.TERMS)}


def events(gained=0, lost=0, place=5, progress=0.0, off=False, sc_phase=0, sc_limit=np.inf, ms_done=False, crashed=False):
    z = jnp.zeros((1, 1))
    i = lambda x: jnp.full((1, 1), x, jnp.int32)
    rec = SimpleNamespace(gained=i(gained), lost=i(lost), place=i(place), under_attack=z > 0, inside_line=z > 0, ms_done=jnp.full((1, 1), ms_done))
    dyn = SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z, slide_front=z, slide_rear=z)
    return SimpleNamespace(
        records=rec, dynamics=dyn, progress=z + progress, crashed=jnp.full((1, 1), crashed), crash_blame=z, black_flag=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, respawned=z > 0,
        off_track=jnp.full((1, 1), off), sc_phase=jnp.full((1,), sc_phase, jnp.int32), sc_limit=jnp.full((1, 1), sc_limit, jnp.float32),
    )


def run(steps, start_place=5, sc_phase=0):
    """steps: (gained, lost, place) per step (or (gained, lost, place, safety car phase)); the total of each term."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), start_place))
    total = np.zeros(len(rw.TERMS))
    for step in steps:
        g, l, p = step[:3]
        ev = events(g, l, p, sc_phase=step[3] if len(step) > 3 else sc_phase)
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
        records=SimpleNamespace(gained=i(0), lost=i(0), place=i(1), under_attack=z > 0, inside_line=z > 0, ms_done=z > 0),
        dynamics=SimpleNamespace(wheelspin=z, lockup_front=z, lockup_rear=z, slide_front=z, slide_rear=z),
        progress=z, crashed=z > 0, crash_blame=z, black_flag=z > 0, warned=i(0), bump=z > 0, pit_in=z > 0, tyre_dead=z > 0, finished=z > 0, tyre_wear=z, off_track=z > 0,
        sc_phase=jnp.zeros(1, jnp.int32), sc_limit=z + np.inf,
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
    assert float(under[0, 0, TERM["tyre_wear"]]) == 0.0  # gentler than rated: no cost (the credit is tyre_save)
    assert np.isclose(float(under[0, 0, TERM["tyre_save"]]), 2e-3) and float(beyond[0, 0, TERM["tyre_save"]]) == 0.0 and float(full[0, 0, TERM["tyre_save"]]) == 0.0
    from f1sim.drivers.config import load

    paid = {p: float(load(p, "laptop").weights[TERM["tyre_save"]]) for p in ("middle", "aggressive", "defensive", "smooth")}
    assert paid == {"middle": 0.0, "aggressive": 0.0, "defensive": 0.0, "smooth": pytest.approx(6.0)}  # x the 0.1 scale
    assert float(load("smooth", "laptop").weights[TERM["tyre_wear"]]) == pytest.approx(-6.0)  # the same rate over the rated wear


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


def test_the_safety_car_counts_as_the_car_ahead():
    """The first car of the queue: closer than 0.5 s behind the safety car is too close, and drawing alongside it is
    charged by the metre, level with it (its black flag) 5.6 m. Not for a lapped car, a car in the pit lane, or with
    the safety car not on the track."""
    from f1env import racecontrol as rc
    from f1env.race import RACING

    bank = SimpleNamespace(length=jnp.array([5000.0]))

    def race(behind, phase=rc.OUT, age=10.0, stage=0, lapped=False):
        """Car 0 `behind` m behind the safety car at 1000 m; car 1 far away (a lap ahead if car 0 is lapped)."""
        return SimpleNamespace(
            status=jnp.full((1, 2), RACING, jnp.int32), pits=SimpleNamespace(stage=jnp.asarray([[stage, 0]], jnp.int32)),
            cars=SimpleNamespace(vx=jnp.full((1, 2), 40.0), vy=jnp.zeros((1, 2))), s=jnp.asarray([[1000.0 - behind, 3000.0]]),
            line=jnp.asarray([[1000.0 - behind, 3000.0 + (5000.0 if lapped else 0.0)]]), tid=jnp.zeros(1, jnp.int32),
            flags=SimpleNamespace(ahead=jnp.zeros((1, 2, 2), bool), slow=jnp.zeros((1, 2))),
            sc=SimpleNamespace(phase=jnp.asarray([phase], jnp.int32), s=jnp.asarray([1000.0]), age=jnp.asarray([age])),
        )

    gap = lambda **k: float(rw.safety_car_gap(bank, race(**k))[0, 0])
    assert gap(behind=30.0) == pytest.approx(30.0) and gap(behind=-2.0) == pytest.approx(-2.0)
    assert all(np.isinf(g) for g in (gap(behind=30.0, phase=rc.CALLED), gap(behind=30.0, age=1.0), gap(behind=30.0, stage=1), gap(behind=30.0, lapped=True), gap(behind=300.0)))
    nobody = jnp.full((1, 2), jnp.inf)
    close = lambda behind: bool(rw.too_close(r := race(behind), nobody, rw.safety_car_gap(bank, r))[0, 0])
    assert close(20.0) and not close(30.0)  # at 40 m/s: 0.5 s is 20 m body to body, 25.6 m centre to centre
    depth = lambda behind: float(rw.alongside(bank, r := race(behind), None, rw.safety_car_gap(bank, r))[0, 0])
    assert depth(10.0) == 0.0 and depth(2.6) == pytest.approx(3.0, abs=1e-4) and depth(0.0) == pytest.approx(5.6, abs=1e-4)
    assert float(rw.alongside(bank, race(0.0))[0, 0]) == 0.0  # without the gap: cars only, as before


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


def test_launch_pays_throttle_with_the_brake_off():
    """In the launch window a step of full throttle earns +20/s x 0.05 s; any brake, no launch window: nothing."""
    ev = events()
    on, f = jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool)
    mem = rw.new_memory(jnp.full((1, 1), 5))
    act = lambda th, br: SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.full((1, 1), th), brake=jnp.full((1, 1), br))
    got = lambda a, launching: float(rw.amounts(ev, a, a, mem, on, f, per_second={"launch": rw.launch_throttle(launching, a)})[0][0, 0, TERM["launch"]])
    assert np.isclose(got(act(1.0, 0.0), on), 0.05) and np.isclose(got(act(0.5, 0.0), on), 0.025)
    assert got(act(1.0, 0.1), on) == 0.0 and got(act(1.0, 0.0), f) == 0.0
    assert rw.weights()[TERM["launch"]] == 20.0


def test_close_calls_are_charged_by_level_and_style():
    ev = events()
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    out, _ = rw.amounts(ev, a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), per_second={"car_close": jnp.full((1, 1), 0.5), "wall_close": jnp.full((1, 1), 1.0)})
    assert np.isclose(float(out[0, 0, TERM["car_close"]]), 0.025) and np.isclose(float(out[0, 0, TERM["wall_close"]]), 0.05)
    w = rw.weights(styles={"car_close": 0.5, "wall_close": 2.0})
    assert w[TERM["car_close"]] == -10.0 and w[TERM["wall_close"]] == -40.0


def test_safety_car_progress_capped_and_no_overtakes():
    """Under the safety car progress pays up to the limit, not past it; places gained from the call to the restart pay
    nothing, then or later; without it both pay as before."""
    from f1env import racecontrol as rc

    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem = rw.new_memory(jnp.full((1, 1), 5))
    lim = 232.0 / 3.6
    prog = lambda v, **kw: float(rw.amounts(events(progress=v * 0.05, **kw), a, a, mem, jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), pace=jnp.full((1, 1), min(lim, 80.0)))[0][0, 0, TERM["progress"]])
    assert prog(80.0, sc_phase=rc.OUT, sc_limit=lim) == pytest.approx(1.0)  # 288 km/h under a 232 limit: one step's worth
    assert prog(50.0, sc_phase=rc.OUT, sc_limit=lim) == pytest.approx(50.0 / lim)
    assert prog(80.0) == pytest.approx(80.0 / lim)  # no limit: all of it
    hold = round(rw.HOLD_S / 0.05)
    gained_under = [(1, 0, 4, rc.OUT)] + [(0, 0, 4, rc.OUT)] * hold + [(0, 0, 4, rc.NONE)] * (hold + 1)
    assert run(gained_under)[TERM["overtake"]] == 0
    assert run([(1, 0, 4)] + [(0, 0, 4)] * (hold + 1))[TERM["overtake"]] == 1
    # a place gained under it, then one after the restart: only the second pays
    assert run(gained_under + [(1, 0, 3)] + [(0, 0, 3)] * (hold + 1))[TERM["overtake"]] == 1


def test_overspeed_charged_by_how_far_over_the_limit():
    """Every step over the safety car's limit costs by the m/s over (from the limit itself), -20 per m/s for a second;
    at or under the limit, without one, or out of the race it's free."""
    from f1env.race import RACING

    lim = 232.0 / 3.6
    race = lambda v, status=RACING: SimpleNamespace(cars=SimpleNamespace(vx=jnp.full((1, 1), v), vy=jnp.zeros((1, 1))), status=jnp.full((1, 1), status, jnp.int32))
    over = lambda v, limit=lim, **kw: float(rw.over_limit(race(v, **kw), jnp.full((1, 1), limit, jnp.float32))[0, 0])
    assert over(lim + 3.0) == pytest.approx(3.0, abs=1e-4)
    assert over(lim - 1.0) == 0.0 and over(lim) == 0.0
    assert over(90.0, limit=np.inf) == 0.0
    assert over(lim + 3.0, status=RACING + 1) == 0.0
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    out, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), per_second={"overspeed": jnp.full((1, 1), 3.0)})
    w = rw.weights()
    assert float(out[0, 0, TERM["overspeed"]] * w[TERM["overspeed"]]) * 20 == pytest.approx(-60.0)  # 3 m/s over for 1 s


def test_the_blue_flag_costs_by_the_second():
    from f1sim.drivers.config import load

    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    one, none = jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool)
    shown, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), one, none, per_second={"blue": one})
    clear, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), one, none, per_second={"blue": none})
    assert float(shown[0, 0, TERM["blue"]]) == pytest.approx(0.05) and float(clear[0, 0, TERM["blue"]]) == 0.0
    assert float(shown[0, 0, TERM["blue"]] * rw.weights()[TERM["blue"]]) * 20 == pytest.approx(-10.0)  # a second of it
    assert all(float(load(p, "laptop").weights[TERM["blue"]]) == pytest.approx(-1.0) for p in ("middle", "smooth"))  # x the 0.1 scale


def test_alongside_the_car_ahead_with_no_passing_costs_by_the_metre():
    """With the safety car out, a car drawing alongside the car ahead of it in the order is charged by how far: nothing
    a car's length back, 5.6 m level, 6.6 m a metre past (the warning's point). Free without the safety car, for a
    lapped car, past a crawling car, or with either car in the pit lane."""
    from f1env import racecontrol as rc
    from f1env.race import RACING

    bank = SimpleNamespace(length=jnp.array([5000.0]))

    def depth(gap, phase=rc.OUT, speed_ahead=50.0, stage=(0, 0), leader=None):
        """Car 0 behind car 1 in the order, `gap` m behind it on the road (negative: past it)."""
        line = [1000.0 - gap, 1000.0] + ([leader] if leader is not None else [])
        n = len(line)
        ahead = np.zeros((1, n, n), bool)
        ahead[0, 1, 0] = True  # car 1 is ahead of car 0 in the order
        race = SimpleNamespace(
            status=jnp.full((1, n), RACING, jnp.int32), pits=SimpleNamespace(stage=jnp.asarray([list(stage) + [0] * (n - 2)], jnp.int32)),
            cars=SimpleNamespace(vx=jnp.asarray([[50.0, speed_ahead] + [50.0] * (n - 2)]), vy=jnp.zeros((1, n))),
            line=jnp.asarray([line], jnp.float32), tid=jnp.zeros(1, jnp.int32),
            flags=SimpleNamespace(ahead=jnp.asarray(ahead), slow=jnp.zeros((1, n))), sc=SimpleNamespace(phase=jnp.asarray([phase], jnp.int32)),
        )
        return float(rw.alongside(bank, race)[0, 0])

    assert depth(10.0) == 0.0 and depth(5.6) == pytest.approx(0.0, abs=1e-4)  # behind it
    assert depth(2.6) == pytest.approx(3.0, abs=1e-4)  # nose 3 m up its side
    assert depth(0.0) == pytest.approx(5.6, abs=1e-4)  # level
    assert depth(-1.0) == pytest.approx(6.6, abs=1e-4) and depth(-9.0) == pytest.approx(6.6, abs=1e-4)  # past: capped
    assert depth(0.0, phase=rc.NONE) == 0.0 and depth(0.0, phase=rc.CALLED) == 0.0  # passing allowed
    assert depth(0.0, phase=rc.GONE) == pytest.approx(5.6, abs=1e-4)  # to the restart
    assert depth(0.0, speed_ahead=3.0) == 0.0  # a crawling car may be passed
    assert depth(0.0, stage=(0, 1)) == 0.0 and depth(0.0, stage=(1, 0)) == 0.0  # the pit lane
    assert depth(0.0, leader=1000.0 + 3000.0) == 0.0  # a lap down: may unlap itself
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    out, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), per_second={"sc_pass": jnp.full((1, 1), 5.6)})
    assert float(out[0, 0, TERM["sc_pass"]] * rw.weights()[TERM["sc_pass"]]) * 20 == pytest.approx(-112.0)  # level for 1 s


def test_pulling_away_from_a_stop_pays_and_stopping_charges_the_same():
    from f1env.pits import OFF
    from f1env.race import RACING

    from f1sim.drivers.config import load

    def race(v, status=RACING, stage=OFF):
        cars = SimpleNamespace(vx=jnp.full((1, 1), v, jnp.float32), vy=jnp.zeros((1, 1), jnp.float32))
        return SimpleNamespace(cars=cars, status=jnp.full((1, 1), status), pits=SimpleNamespace(stage=jnp.full((1, 1), stage)))

    pull = lambda a, b, **kw: float(rw.pull_away(race(a, **kw), race(b, **kw))[0, 0])
    assert pull(0.0, 0.5) == pytest.approx(0.5)  # the first step of throttle from rest
    assert pull(0.5, 3.0) == pytest.approx(0.5)  # out of the band: only the part inside it
    assert pull(3.0, 0.0) == pytest.approx(-1.0)  # slowing into it costs the same back
    assert pull(5.0, 9.0) == 0.0 and pull(0.0, 0.0) == 0.0  # above the band, or standing: nothing
    assert pull(0.0, 0.5, status=RACING + 1) == 0.0 and pull(0.0, 0.5, stage=OFF + 1) == 0.0  # out, or in the pit lane
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    out, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), per_step={"pull_away": jnp.full((1, 1), 0.5)})
    assert float(out[0, 0, TERM["pull_away"]] * rw.weights()[TERM["pull_away"]]) == pytest.approx(50.0)
    w = load("smooth", "laptop").weights[TERM["pull_away"]]
    assert float(w) == pytest.approx(10.0)  # x the 0.1 scale: +5 scaled for 0.5 m/s, past the 3.2 its critics put on staying
    # a stop and a start come to nothing: 3 -> 0 -> 1 m/s in steps
    speeds = [3.0, 2.0, 0.6, 0.0, 0.0, 0.4, 0.9, 1.5]
    assert sum(pull(x, y) for x, y in zip(speeds, speeds[1:])) == pytest.approx(0.0, abs=1e-6)


def test_a_waiting_car_pays_a_deposit_and_gets_it_back_by_driving_off():
    from f1env.pits import OFF
    from f1env.race import RACING, RaceConfig

    from f1sim.drivers.config import load

    cfg = RaceConfig(stuck_s=90.0)  # one stuck time: the timer is in real seconds, no track lookup

    def race(v, waited, status=RACING, stage=OFF):
        cars = SimpleNamespace(vx=jnp.full((1, 1), v, jnp.float32), vy=jnp.zeros((1, 1), jnp.float32), x=jnp.zeros((1, 1)), y=jnp.zeros((1, 1)))
        return SimpleNamespace(cars=cars, status=jnp.full((1, 1), status), pits=SimpleNamespace(stage=jnp.full((1, 1), stage)), stuck_time=jnp.full((1, 1), waited, jnp.float32), tid=jnp.zeros(1, jnp.int32))

    wait = lambda a, b: float(rw.waiting(None, race(*a), race(*b), cfg)[0, 0])
    level = lambda t: 1.0 - np.exp(-t / 4.0)
    assert wait((0.0, 0.0), (0.0, 4.0)) == pytest.approx(-level(4.0))  # 4 s standing: 63% of the deposit paid in
    assert wait((0.0, 4.0), (0.0, 8.0)) == pytest.approx(level(4.0) - level(8.0))  # the next 4 s: 23% more
    assert wait((0.0, 60.0), (0.0, 64.0)) == pytest.approx(0.0, abs=1e-5)  # a long wait: all paid, nothing more
    assert wait((0.0, 8.0), (1.5, 8.0)) == pytest.approx(level(8.0))  # it drives off: all of it back at once
    assert wait((0.0, 8.0), (0.5, 8.0)) == pytest.approx(0.5 * level(8.0))  # half-way through the stopped band: half
    assert wait((5.0, 8.0), (6.0, 9.0)) == 0.0  # moving without progress (the timer runs): no deposit
    assert wait((0.0, 8.0), (0.0, 8.0)) == 0.0  # queuing (the timer paused): nothing changes
    assert wait((0.0, 8.0), (0.0, 0.0)) == pytest.approx(level(8.0))  # the timer reset by progress under 1 m/s: back too
    assert wait((0.0, 8.0), (0.0, 8.05, RACING + 1)) == 0.0  # timed out or hit while waiting: the deposit is lost
    assert wait((0.0, 8.0, RACING, OFF + 1), (0.0, 8.05, RACING, OFF + 1)) == 0.0  # the pit lane
    # a stop, a wait and a start come to nothing
    steps = [(3.0, 0.0), (0.4, 0.0), (0.0, 1.0), (0.0, 6.0), (0.0, 20.0), (0.6, 20.0), (2.0, 20.0), (9.0, 0.0)]
    assert sum(wait(a, b) for a, b in zip(steps, steps[1:])) == pytest.approx(0.0, abs=1e-6)
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    out, _ = rw.amounts(events(), a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), per_step={"waiting": jnp.full((1, 1), -1.0)})
    assert float(out[0, 0, TERM["waiting"]] * rw.weights()[TERM["waiting"]]) == pytest.approx(-2000.0)
    assert float(load("smooth", "laptop").weights[TERM["waiting"]]) == pytest.approx(200.0)  # x the 0.1 scale: most of a crash (250)


def _crash_step(left=None, black=False):
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    ev = events()
    if black:
        ev.black_flag = jnp.ones((1, 1), bool)
    else:
        ev.crashed = jnp.ones((1, 1), bool)
    out, _ = rw.amounts(ev, a, a, rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool), left=left)
    return float(out[0, 0, TERM["black_flag" if black else "crash"]])


def test_a_crash_costs_more_the_more_race_it_throws_away():
    assert _crash_step() == 1.0 and _crash_step(jnp.zeros((1, 1))) == 1.0
    assert np.isclose(_crash_step(jnp.full((1, 1), 3.0)), 4.0) and np.isclose(_crash_step(jnp.full((1, 1), 3.0), black=True), 4.0)
    tid = jnp.zeros((2,), jnp.int32)
    bank = SimpleNamespace(length=jnp.asarray([5000.0]))
    race = SimpleNamespace(target_laps=jnp.asarray([20, 20]), tid=tid, line=jnp.asarray([[1000.0, 99000.0], [0.0, 0.0]]), session=jnp.asarray([0, 1]))
    left = np.asarray(rw.race_left(bank, race, 0.059))
    assert np.isclose(left[0, 0], 0.059 * 99.0) and np.isclose(left[0, 1], 0.059 * 1.0)  # lap 1 vs the last km
    assert (left[1] == 0.0).all()  # qualifying: no race to lose
    long = SimpleNamespace(target_laps=jnp.asarray([60]), tid=jnp.zeros((1,), jnp.int32), line=jnp.zeros((1, 1)), session=jnp.asarray([0]))
    assert float(rw.race_left(bank, long, 0.059)[0, 0]) == rw.LEFT_CAP  # 300 km left: capped
    from f1sim.drivers.config import load

    assert load("smooth", "laptop").env.crash_left_km == 0.0  # off since 2026-11-17 (run s9 got worse with it)


def test_no_tyre_save_behind_the_safety_car():
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    mem, on, f = rw.new_memory(jnp.full((1, 1), 5)), jnp.ones((1, 1), bool), jnp.zeros((1, 1), bool)
    paid = {}
    for phase in (0, 1, 2):
        ev = events(sc_phase=phase)
        ev.tyre_wear = jnp.full((1, 1), 1e-5)
        out, _ = rw.amounts(ev, a, a, mem, on, f, rated_wear=jnp.full((1, 1), 5e-5))
        paid[phase] = float(out[0, 0, TERM["tyre_save"]])
    assert np.isclose(paid[0], 4e-3) and paid[1] == 0.0 and paid[2] == 0.0


def test_time_is_charged_at_each_mini_sector_end_for_the_racing_steps_it_took():
    """Not in qualifying or under the red flag; the steps since the last mini-sector are charged when the next one
    ends (or the car's race does): a faster car pays less, sooner."""
    a = SimpleNamespace(steer=jnp.zeros((1, 1)), throttle=jnp.zeros((1, 1)), brake=jnp.zeros((1, 1)))
    inf = jnp.full((1, 1), jnp.inf)

    def charges(steps):  # steps: (qualifying, safety car phase, mini-sector ends, crashed)
        mem, out = rw.new_memory(jnp.full((1, 1), 5)), []
        for is_q, phase, done, crash in steps:
            ev = events(sc_phase=phase, ms_done=done, crashed=crash)
            amt, mem = rw.amounts(ev, a, a, mem, jnp.ones((1, 1), bool), ev.respawned, quali=(jnp.array([is_q]), inf, inf))
            out.append(float(amt[0, 0, TERM["time"]]))
        return out

    assert charges([(False, 0, False, False)] * 3 + [(False, 0, True, False)]) == [0, 0, 0, 4]  # 4 steps, paid at the end
    assert charges([(False, 0, True, False)] * 2) == [1, 1]  # a faster car: less each time
    assert charges([(False, 5, False, False), (False, 2, False, False), (True, 0, False, False), (False, 0, True, False)]) == [0, 0, 0, 2]  # red flag, qualifying free
    assert charges([(False, 0, False, False), (False, 0, False, True)]) == [0, 2]  # the rest when its race ends
    assert rw.BASE["time"] == 0.0  # off unless a profile sets it (Smooth doesn't)


def test_every_warning_kind_counts_towards_the_warning_term():
    for bit in range(7):  # speeding, passing, blue, slow, blend, close, far
        assert float(rw.warning_shares(jnp.array([[1 << bit]]), jnp.array([[0]]))[0, 0]) == 0.25, bit


def test_order_terms_follow_the_compliance_records():
    from f1env.race import RACING

    from f1env import orders as od
    from f1env.pits import OFF

    one = jnp.ones((1, 2), bool)
    rec = SimpleNamespace(push_met=jnp.array([[True, False]]), save_met=~one, push_missed=jnp.array([[False, True]]), save_missed=~one,
                          let_past_failed=jnp.array([[True, False]]), passed_mate_on_hold=~one, lost=jnp.array([[0, 1]]))
    race = SimpleNamespace(status=jnp.full((1, 2), RACING), pits=SimpleNamespace(stage=jnp.full((1, 2), OFF)), battery=jnp.full((1, 2), 5.0),
                           ers_enabled=jnp.array([True]), orders=SimpleNamespace(team=jnp.array([[od.LET_PAST, od.DEFEND]]), ers=jnp.array([[od.ERS_SAVE, od.DEPLOY]])))
    rec.ers_on_save = jnp.array([[True, False]])  # deploying under "save"
    ev = SimpleNamespace(records=rec, ers_on=jnp.array([[True, False]]))
    acts = SimpleNamespace(throttle=jnp.ones((1, 2)))
    got = {k: np.asarray(v)[0] for k, v in rw.order_steps(race, ev, acts, 0.05).items()}
    assert list(got["order_pace"]) == [1.0, -1.0] and list(got["order_team"]) == [1.0, 0.0] and list(got["order_defend"]) == [0.0, 1.0]
    assert np.allclose(got["order_ers"], [0.05, 0.05])  # deploying on save, idle on deploy at full throttle
    w = dict(zip(rw.TERMS, rw.weights()))
    assert w["order_pace"] == 20.0 and w["order_team"] == -500.0 and w["order_defend"] == -300.0 and w["order_ers"] == -5.0
