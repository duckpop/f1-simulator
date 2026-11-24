import jax
import numpy as np

from f1sim import tqc
from f1sim.drivers import evaluate
from f1sim.drivers.config import load


def test_alone_runs_every_race_to_its_end(bank):
    cfg = load("middle").tqc
    agent = tqc.init(jax.random.PRNGKey(0), cfg)  # untrained: it won't get far
    r = evaluate.alone(bank, agent.actor, cfg, laps=1, chunk=1000, weights=load("middle").weights)
    assert r["races"] == 2 and 0.0 <= r["finish_rate"] <= 1.0 and 0.0 <= r["distance"] <= 1.0
    assert r["steps"] < 30_000  # stopped once both races were over (the time limit at the latest)
    assert np.isfinite([r["crashes_per_lap"], r["lap_ratio"], r["laps_per_stint"], r["reward_per_step"]]).all()
    assert r["eval_version"] == evaluate.EVAL_VERSION
    assert np.isnan(evaluate.alone(bank, agent.actor, cfg, laps=1, chunk=1000)["reward_per_step"])  # no weights: not measured


def test_better_prefers_distance_then_the_profiles_reward():
    a = {"eval_version": evaluate.EVAL_VERSION, "distance": 0.80, "reward_per_step": 0.50}
    assert evaluate.better(a, None)
    assert evaluate.better(a, {**a, "distance": 0.75, "reward_per_step": 0.9})  # 5% more of the distance
    assert not evaluate.better(a, {**a, "distance": 0.85, "reward_per_step": 0.1})
    assert evaluate.better(a, {**a, "distance": 0.81, "reward_per_step": 0.45})  # within 2%: the reward decides
    assert not evaluate.better(a, {**a, "distance": 0.79, "reward_per_step": 0.55}) and not evaluate.better(a, a)
    assert evaluate.better(a, {"finish_rate": 1.0, "laps": 500.0, "crashes_per_lap": 0.0, "lap_ratio": 1.0})  # an older eval is no bar


def test_stage_b_prefers_distance_then_points_then_reward():
    a = {"eval_version": evaluate.EVAL_VERSION, "points_per_race": 5.0, "distance": 0.8, "reward_per_step": 0.5}
    assert not evaluate.better_ordered(a, {**a, "points_per_race": 4.5, "distance": 0.95})  # the distance before the points
    assert evaluate.better_ordered(a, {**a, "points_per_race": 9.0, "distance": 0.7})
    assert evaluate.better_ordered(a, {**a, "points_per_race": 4.5, "distance": 0.81})  # within 2%: the points
    assert evaluate.better_ordered(a, {**a, "points_per_race": 4.95, "reward_per_step": 0.4}) and not evaluate.better_ordered(a, a)
    assert evaluate.better_ordered(a, {"points_per_race": 9.0, "finish_rate": 1.0, "crashes_per_lap": 0.0})  # an older eval


def test_incidents_per_100km_with_an_interval():
    import json

    r = evaluate.incidents(np.float32(10), np.float32(500.0))  # numpy inputs, as the tallies give them
    json.dumps(r)
    assert np.isclose(r["incidents_per_100km"], 2.0)
    lo, hi = r["incidents_ci"]
    assert 0.9 < lo < 2.0 < hi < 3.8  # the exact Poisson interval for 10 is 4.80-18.39 events
    assert evaluate.incidents(0, 100.0)["incidents_ci"][0] == 0.0


def test_the_safer_driver_wins_beyond_the_noise():
    base = {"eval_version": evaluate.EVAL_VERSION, "points_per_race": 5.0, "distance": 0.5, "reward_per_step": 0.05}
    safe = {**base, **evaluate.incidents(5, 1000.0), "points_per_race": 2.0}
    risky = {**base, **evaluate.incidents(40, 1000.0), "points_per_race": 8.0}
    assert evaluate.safer(safe, risky) == -1 and evaluate.better_ordered(safe, risky) and not evaluate.better_ordered(risky, safe)
    close = {**base, **evaluate.incidents(6, 1000.0), "points_per_race": 6.0}
    assert evaluate.safer(close, safe) == 0 and evaluate.better_ordered(close, safe)  # within the noise (same distance): the points


TYRES = ("incidents_vs_best", "tyres", "distance", "points")  # Smooth's order before 2026-11-21 (best_by_tyres)


def test_a_tyre_profile_weighs_tyre_life_after_incidents():
    base = {"eval_version": evaluate.EVAL_VERSION, "points_per_race": 5.0, "distance": 0.8, "reward_per_step": 0.05, **evaluate.incidents(20, 10000.0)}
    long_life = {**base, "km_per_tyre_set": 210.0, "points_per_race": 3.0, "distance": 0.7}
    short_life = {**base, "km_per_tyre_set": 180.0}
    assert evaluate.better_ordered(long_life, short_life, TYRES) and not evaluate.better_ordered(short_life, long_life, TYRES)
    assert not evaluate.better_ordered(long_life, short_life)  # without it the distance decides, as before
    near = {**short_life, "km_per_tyre_set": 185.0, "distance": 0.85}
    assert evaluate.better_ordered(near, short_life, TYRES)  # within 5%: on to the distance
    crashy = {**long_life, "km_per_tyre_set": 300.0, **evaluate.incidents(60, 10000.0)}
    assert not evaluate.better_ordered(crashy, short_life, TYRES)  # incidents still come first


def test_clean_running_counts_only_a_clear_road():
    from types import SimpleNamespace
    import jax.numpy as jnp
    from f1sim.drivers import env as de
    one = lambda x: jnp.array([[x]])
    def ev_(gap=1e9, off=False, pit=False, phase=0, progress=4.0):
        return SimpleNamespace(progress=one(progress), gap_ahead=one(gap), off_track=one(off), in_pit=one(pit), sc_phase=jnp.array([phase]))
    t, sim, _ = de.clean_running(ev_(), one(True), one(40.0), one(False), one(False))
    assert abs(float(t[0, 0]) - de.STEP_S) < 1e-6 and abs(float(sim[0, 0]) - 0.1) < 1e-6  # 4 m at the lap sim's 40 m/s: 0.1 s
    for e, launch, blue, valid in ((ev_(gap=50.0), False, False, True), (ev_(off=True), False, False, True), (ev_(pit=True), False, False, True),
                                   (ev_(phase=1), False, False, True), (ev_(), True, False, True), (ev_(), False, True, True), (ev_(), False, False, False)):
        assert float(de.clean_running(e, one(valid), one(40.0), one(launch), one(blue))[0][0, 0]) == 0.0  # 50 m at 80 m/s: traffic


def test_profiles_are_set_against_the_reference_field():
    from f1sim.drivers import season
    res = {"profiles": {"middle": {"clean_pace": 1.02, "incidents": 30, "km": 10_000}, "reference": {"clean_pace": 1.05, "incidents": 10, "km": 10_000}}}
    r = season.against_reference(res)["profiles"]["middle"]
    assert abs(r["pace_vs_reference"] - 1.02 / 1.05) < 1e-9 and r["riskier_than_reference"]


def _res(pace=1.0, inc=10, ref_inc=10, km=10_000, held=0.9, won=0.2, dist=0.9, pts=5.0, tyres=200.0):
    return {"eval_version": evaluate.EVAL_VERSION, "incidents": inc, "km": km, "reference_incidents": ref_inc, "reference_km": km, "pace_vs_reference": pace,
            "attacks_held": held, "attack_success": won, "distance": dist, "points_per_race": pts, "reward_per_step": 0.0, "km_per_tyre_set": tyres}


def test_each_profile_judges_a_best_in_its_own_order():
    import tomllib
    cfg = tomllib.load(open("configs/drivers.toml", "rb"))
    orders = {**cfg["stage_d"]["best_order"], "middle": cfg["stage_b"]["best_order"]["middle"]}  # the personalities' are stage D's
    best = _res()
    # Aggressive: pace first, whatever the incidents; within the pace band, not riskier than the reference, then attacks
    assert evaluate.better_ordered(_res(pace=0.99, inc=150, ref_inc=50, km=50_000), _res(km=50_000, inc=50, ref_inc=50), orders["aggressive"])
    assert not evaluate.better_ordered(_res(pace=0.999, inc=150, ref_inc=50, km=50_000, won=0.5), _res(km=50_000, inc=50, ref_inc=50), orders["aggressive"])
    assert evaluate.better_ordered(_res(won=0.3), best, orders["aggressive"])
    # Middle: the same faster car with 1.6x is clearly riskier than the reference: no
    assert not evaluate.better_ordered(_res(pace=0.99, inc=16, ref_inc=4, km=4000), _res(km=4000, inc=4, ref_inc=4), orders["middle"])
    assert evaluate.better_ordered(_res(pace=0.99), best, orders["middle"])
    # Defensive: attacks held first, then incidents, pace; points at the end
    assert evaluate.better_ordered(_res(held=0.95, pace=1.02), best, orders["defensive"])
    assert evaluate.better_ordered(_res(held=0.95, inc=150, ref_inc=50, km=50_000), _res(km=50_000, inc=50, ref_inc=50), orders["defensive"])
    assert not evaluate.better_ordered(_res(held=0.91, inc=150, ref_inc=50, km=50_000), _res(km=50_000, inc=50, ref_inc=50), orders["defensive"])
    assert evaluate.better_ordered(_res(pts=6.0), best, orders["defensive"])
    # Smooth: tyres right after incidents against the best
    assert evaluate.better_ordered(_res(tyres=220.0, pace=1.05), best, orders["smooth"])


def test_riskier_scales_the_reference_rate():
    r = _res(inc=30, ref_inc=10, km=10_000)
    assert evaluate.riskier(r, 1.0) and not evaluate.riskier(r, 2.0)  # 3x: riskier than 1x; against 2x, within the noise of 10 incidents
    assert evaluate.riskier(_res(inc=150, ref_inc=50, km=50_000), 2.0)  # 3x with eval-sized counts: riskier than 2x
    assert evaluate.riskier({"incidents": 1, "km": 1}) is None  # no reference field
