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


def test_stage_b_prefers_points_against_the_reference_then_distance():
    a = {"eval_version": evaluate.EVAL_VERSION, "points_per_race": 5.0, "distance": 0.8, "reward_per_step": 0.5}
    assert evaluate.better_mixed(a, {**a, "points_per_race": 4.5, "distance": 0.95})
    assert evaluate.better_mixed(a, {**a, "points_per_race": 5.05, "distance": 0.7})  # within 0.1 points: the distance
    assert evaluate.better_mixed(a, {**a, "points_per_race": 4.95, "reward_per_step": 0.4}) and not evaluate.better_mixed(a, a)
    assert evaluate.better_mixed(a, {"points_per_race": 9.0, "finish_rate": 1.0, "crashes_per_lap": 0.0})  # an older eval


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
    assert evaluate.safer(safe, risky) == -1 and evaluate.better_mixed(safe, risky) and not evaluate.better_mixed(risky, safe)
    close = {**base, **evaluate.incidents(6, 1000.0), "points_per_race": 6.0}
    assert evaluate.safer(close, safe) == 0 and evaluate.better_mixed(close, safe)  # within the noise: the points decide
