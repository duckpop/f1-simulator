import jax
import numpy as np

from f1sim import tqc
from f1sim.drivers import evaluate
from f1sim.drivers.config import load


def test_alone_runs_every_race_to_its_end(bank):
    cfg = load("middle").tqc
    agent = tqc.init(jax.random.PRNGKey(0), cfg)  # untrained: it won't get far
    r = evaluate.alone(bank, agent.actor, cfg, laps=1, chunk=1000)
    assert r["races"] == 2 and 0.0 <= r["finish_rate"] <= 1.0
    assert r["steps"] < 30_000  # stopped once both races were over (the time limit at the latest)
    assert np.isfinite([r["crashes_per_lap"], r["lap_ratio"], r["laps_per_stint"]]).all()


def test_better_prefers_finishing_then_safety_then_pace():
    a = {"finish_rate": 0.6, "crashes_per_lap": 0.02, "lap_ratio": 1.4}
    assert evaluate.better(a, None)
    assert evaluate.better(a, {**a, "finish_rate": 0.5})
    assert evaluate.better(a, {**a, "crashes_per_lap": 0.03})
    assert evaluate.better(a, {**a, "lap_ratio": 1.5}) and not evaluate.better(a, a)
