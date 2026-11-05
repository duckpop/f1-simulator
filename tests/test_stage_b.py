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
    # 200 steps: the cars are held at the lights for the first 5-8 s, and nothing is learned from a batch with no
    # step the driver controlled
    chunk = stage_b.make_chunk(cfgs, env_cfg, seats, weights, steps=200, updates=1, learn_from=10)
    before = [jax.tree.map(jnp.copy, a.actor) for a in agents]  # the chunk donates the agents
    league, (stats, metrics) = chunk(league, bank)
    assert int(league.steps) == 200 and all(int(r.size) == 64 for r in league.replays)
    assert stats["amounts"].shape == (200, 5, len(rewards.TERMS)) and metrics["q"].shape == (200, 4)
    assert float(np.abs(np.asarray(metrics["q"][:100])).max()) == 0.0  # before the lights: no valid batch, no update
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
    print(evaluate.summary_mixed(r["profiles"]["middle"]))
    # with a reference field in the odd cars and the profiles' weights
    ref = tqc.init(jax.random.PRNGKey(9), tiny(2)).actor
    from f1sim.drivers.config import load

    w = [load(p).weights for p in ("middle", "aggressive", "smooth", "defensive")]
    r = season.eval_weekend(env, actors, tiny(2), weights=w, reference=ref)
    assert set(r["profiles"]) == {"middle", "aggressive", "smooth", "defensive", "reference"}
    for p in r["profiles"].values():
        assert 0 <= p["distance"] <= 1 and p["eval_version"] == evaluate.EVAL_VERSION
    assert np.isnan(r["profiles"]["reference"]["reward_per_step"]) and np.isfinite(r["profiles"]["middle"]["reward_per_step"])


def test_stored_seats_take_a_share_of_a_profiles_seats():
    seats = stage_b.seat_layout(22, 16, 1)  # one profile: 16 learning, 6 frozen
    assert list(stage_b.stored_seats(seats, 0, 0.25)) == [0, 1, 2, 3]
    assert len(stage_b.stored_seats(seats, 0)) == 16
    four = stage_b.seat_layout(22, 4)
    assert len(stage_b.stored_seats(four, 2, 0.25)) == 1  # at least one


def test_a_race_restarts_once_its_learners_are_out():
    """With `learners`, a race is over when none of those cars is going, whatever the others do."""
    from types import SimpleNamespace

    from f1env.race import OUT, RACING, RaceConfig

    bank = SimpleNamespace(length=jnp.array([5000.0]))
    status = jnp.array([[OUT, OUT, RACING]])  # the two learners are out, a frozen car runs on
    race = SimpleNamespace(
        status=status, dnf=status == OUT, flags=SimpleNamespace(black=jnp.zeros((1, 3), bool)), line=jnp.zeros((1, 3)), tid=jnp.array([0]),
        target_laps=jnp.array([5]), clock=jnp.array([10.0]), flag=jnp.array([False]),
    )
    limit = jnp.array([1e6])
    no_respawn = RaceConfig(respawn=False)
    assert not bool(de.race_done(bank, race, limit, de.EnvConfig(cars=3, race=no_respawn))[0])
    assert bool(de.race_done(bank, race, limit, de.EnvConfig(cars=3, race=no_respawn, learners=(True, True, False)))[0])
