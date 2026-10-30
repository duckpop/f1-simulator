from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1sim import replay as rp
from f1sim import tqc
from f1sim.drivers import env as de
from f1sim.drivers import evaluate, rewards, stage_b
from f1sim.nets import EncoderSpec

TINY = dict(encoder=EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,)), critics=2, quantiles=3, actor_hidden=(8,), critic_hidden=(8,), batch=8, n_step=2)


def tiny(drop):
    return tqc.TQCConfig(obs_dim=de.OBS, act_dim=de.ACTIONS, drop=drop, **TINY)


def test_seats():
    seats = stage_b.seat_layout(22, 4)
    assert [int((seats == p).sum()) for p in range(4)] == [4, 4, 4, 4] and int((seats < 0).sum()) == 6
    assert all(seats[2 * t] != seats[2 * t + 1] for t in range(8))  # team-mates are different profiles
    with pytest.raises(ValueError):
        stage_b.seat_layout(22, 6)


def test_init_paths():
    inits = {"smooth": "runs/middle/stage-b/best.npz"}
    assert stage_b.init_path("smooth", "runs", inits) == "runs/middle/stage-b/best.npz"
    assert stage_b.init_path("smooth", "runs", inits, resume=True, init_all="x.npz") == "runs/middle/stage-b/best.npz"
    assert stage_b.init_path("middle", "runs", inits) == "runs/middle/stage-a/best.npz"
    assert stage_b.init_path("middle", "runs", inits, resume=True) == "runs/middle/stage-b/latest.npz"
    assert stage_b.init_path("middle", "runs", inits, init_all="x.npz") == "x.npz"


def test_league_chunk(bank):
    cfgs = [tiny(d) for d in (2, 1, 1, 2)]  # different truncations compile side by side
    seats = stage_b.seat_layout(22, 4)
    env_cfg = de.EnvConfig(races=1, cars=22, quali_share=0.0)
    agents = [tqc.init(jax.random.PRNGKey(p), c) for p, c in enumerate(cfgs)]
    replays = [rp.new(64, 4, de.OBS, de.ACTIONS) for _ in cfgs]
    pool = stage_b.stack([agents[k % 4].actor for k in range(3)])
    st, obs = jax.jit(partial(de.reset, cfg=env_cfg))(bank, jax.random.PRNGKey(0))
    league = stage_b.League(st, obs, tuple(agents), tuple(replays), pool, jnp.zeros((1, 6), jnp.int32), jax.random.PRNGKey(1), jnp.zeros((), jnp.int32))
    weights = np.stack([rewards.weights() for _ in cfgs])
    chunk = stage_b.make_chunk(cfgs, env_cfg, seats, weights, steps=20, updates=1, learn_from=10)
    before = [jax.tree.map(jnp.copy, a.actor) for a in agents]  # the chunk donates the agents
    league, (stats, metrics) = chunk(league, bank)
    assert int(league.steps) == 20 and all(int(r.size) == 20 for r in league.replays)
    assert stats["amounts"].shape == (20, 5, len(rewards.TERMS)) and metrics["q"].shape == (20, 4)
    moved = lambda x, y: any(float(jnp.max(jnp.abs(a - b))) > 0 for a, b in zip(jax.tree.leaves(x), jax.tree.leaves(y)))
    assert all(moved(b, a.actor) for b, a in zip(before, league.agents))  # every profile learned
    assert int(league.pool_of.max()) < 3
    league = stage_b.push_snapshot(league, 1, league.agents[2].actor)
    for leaf_pool, leaf_actor in zip(jax.tree.leaves(league.pool), jax.tree.leaves(league.agents[2].actor)):
        np.testing.assert_array_equal(np.asarray(leaf_pool[1]), np.asarray(leaf_actor))


@pytest.mark.slow
def test_eval_weekend(tracks):
    from f1env.env import F1Env
    from f1env.weekend import WeekendConfig

    from f1sim.drivers import season

    actors = [tqc.init(jax.random.PRNGKey(p), tiny(2)).actor for p in range(4)]
    env = F1Env(tracks, WeekendConfig(race_laps=(1, 1), quali_sessions=((120.0, 6), (120.0, 6), (120.0, 0))))
    r = season.eval_weekend(env, actors, tiny(2))
    assert set(r["profiles"]) == {"middle", "aggressive", "smooth", "defensive"}
    for p in r["profiles"].values():
        assert 0 <= p["finish_rate"] <= 1 and 0 <= p["out_lap1"] <= 1 and p["points_per_race"] >= 0
    assert sum(r["profiles"][p]["mean_grid_slot"] * n for p, n in zip(r["profiles"], (6, 6, 5, 5))) == pytest.approx(sum(range(1, 23)))  # the grid is qualifying's
    assert evaluate.better_mixed({"points_per_race": 3.0, "finish_rate": 0.5, "crashes_per_lap": 0.1}, {"points_per_race": 2.0, "finish_rate": 0.9, "crashes_per_lap": 0.0})
    print(evaluate.summary_mixed(r["profiles"]["middle"]))
