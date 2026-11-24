from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from f1sim import loop
from f1sim import replay as rp
from f1sim import nets, tqc
from f1sim.drivers import env as de
from f1sim.drivers import evaluate, rewards, stage_b
from f1sim.nets import EncoderSpec

TINY = dict(encoder=EncoderSpec(lidar_beams=50, conv=((4, 5, 5),), other_hidden=(8,)), critics=2, quantiles=3, actor_hidden=(8,), critic_hidden=(8,), batch=8, n_step=2)


def tiny(drop):
    return tqc.TQCConfig(obs_dim=de.OBS, act_dim=de.ACTIONS, drop=drop, **TINY)


def test_every_snapshot_drives_an_equal_share_of_the_frozen_cars():
    a = np.asarray(stage_b.assign_pool(jax.random.PRNGKey(0), races=16, frozen=16, size=8))
    assert a.shape == (16, 16) and a.dtype == np.int32
    for row in a:  # every race: each of the 8 snapshots in exactly 2 of the 16 frozen cars
        np.testing.assert_array_equal(np.bincount(row, minlength=8), np.full(8, 2))
    assert len({tuple(row) for row in a}) > 1  # shuffled, differently per race
    b = np.asarray(stage_b.assign_pool(jax.random.PRNGKey(1), races=4, frozen=6, size=4))
    for row in b:  # 6 cars, 4 snapshots: two snapshots get 2 cars, two get 1
        assert sorted(np.bincount(row, minlength=4)) == [1, 1, 2, 2]


def test_seats():
    seats = stage_b.seat_layout(22, 4)
    assert [int((seats == p).sum()) for p in range(4)] == [4, 4, 4, 4] and int((seats < 0).sum()) == 6
    assert all(seats[2 * t] != seats[2 * t + 1] for t in range(8))  # team-mates are different profiles
    with pytest.raises(ValueError):
        stage_b.seat_layout(22, 6)


def test_init_paths(tmp_path):
    """(An empty runs directory: the repo's own runs/ may hold a together best, which stage B would start from.)"""
    inits = {"smooth": "runs/middle/stage-b/best.npz"}
    assert stage_b.init_path("smooth", tmp_path, inits) == "runs/middle/stage-b/best.npz"
    assert stage_b.init_path("smooth", tmp_path, inits, resume=True, init_all="x.npz") == "runs/middle/stage-b/best.npz"
    assert stage_b.init_path("middle", tmp_path, inits) == str(tmp_path / "middle" / "stage-a" / "best.npz")
    assert stage_b.init_path("middle", tmp_path, inits, resume=True) == str(tmp_path / "middle" / "stage-b" / "latest.npz")
    assert stage_b.init_path("middle", tmp_path, inits, init_all="x.npz") == "x.npz"


def test_init_paths_follow_the_regime(tmp_path):
    """A -> B -> C (Middle) -> D (every personality from Middle's C best); each resumes its own latest."""
    assert stage_b.init_path("middle", tmp_path, {}) == str(tmp_path / "middle" / "stage-a" / "best.npz")
    assert stage_b.init_path("middle", tmp_path, {}, stage="c") == str(tmp_path / "middle" / "stage-b" / "best.npz")
    assert stage_b.init_path("smooth", tmp_path, {}, stage="d") == str(tmp_path / "middle" / "stage-c" / "best.npz")
    assert stage_b.init_path("smooth", tmp_path, {}, resume=True, stage="d") == str(tmp_path / "smooth" / "stage-d" / "latest.npz")


def test_stages_b_and_c_train_middle_alone():
    with pytest.raises(SystemExit):
        stage_b.main(["--stage", "c", "--generate", "2", "--profiles", "middle,smooth"])


def test_league_chunk(bank):
    cfgs = [tiny(d) for d in (2, 1, 1, 2)]  # different truncations compile side by side
    seats = stage_b.seat_layout(22, 4)
    env_cfg = de.EnvConfig(races=1, cars=22, quali_share=0.0)
    agents = [tqc.init(jax.random.PRNGKey(p), c) for p, c in enumerate(cfgs)]
    replays = [rp.new(64, 4, de.OBS, de.ACTIONS, de.storage_codec()) for _ in cfgs]  # the training's packing, shared by the replays
    pool = stage_b.stack([agents[k % 4].actor for k in range(3)])
    st, obs = jax.jit(partial(de.reset, cfg=env_cfg))(bank, jax.random.PRNGKey(0))
    league = stage_b.League(st, obs, tuple(agents), tuple(replays), pool, jnp.zeros((1, 6), jnp.int32), jax.random.PRNGKey(1), jnp.zeros((), jnp.int32))
    weights = np.stack([rewards.weights() for _ in cfgs])
    # 200 steps: the cars are held at the lights for the first 5-8 s, and nothing is learned from a batch with no
    # step the driver controlled
    chunk = stage_b.make_chunk(cfgs, env_cfg, seats, weights, steps=200, updates=1, plan=loop.Schedule(10, 10, 10), stride=2)  # a gradient step every 2 env steps
    before = [jax.tree.map(jnp.copy, a.actor) for a in agents]  # the chunk donates the agents
    league, (stats, metrics) = chunk(league, bank)
    assert int(league.steps) == 200 and all(int(r.size) == 64 for r in league.replays)
    assert stats["amounts"].shape == (200, 5, len(rewards.TERMS)) and metrics["q"].shape == (200, 4)
    assert float(np.abs(np.asarray(metrics["q"][:100])).max()) == 0.0  # before the lights: no valid batch, no update
    learning = np.asarray(metrics["learning"])
    assert learning[11::2].max() == 0.0 and learning[10::2].min() == 1.0  # every 2nd step learns (stride 2)
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


def test_updates_follow_the_reuse():
    from f1sim.drivers.stage_b import updates_for

    assert updates_for({"reuse": 32}, 1, 64, 256) == 8  # 16 races x 4 stored seats, batch 256
    assert updates_for({"reuse": 32}, 1, 256, 256) == 32
    assert updates_for({}, 3, 64, 256) == 3
    assert updates_for({"reuse": 1}, 1, 64, 256) == 1  # at least one


def test_learn_every_spaces_out_the_gradient_steps():
    from f1sim.drivers.stage_b import learn_every
    assert learn_every({"reuse": 1.33}, 96, 256) == 2  # 6 stored seats x 16 races: a gradient step every 2 env steps
    assert learn_every({"reuse": 4}, 64, 256) == 1
    assert learn_every({}, 64, 256) == 1


def test_warm_start_schedule():
    """From scratch: learn as soon as a batch can be drawn, everything at once. From a checkpoint: fill 20% of the
    replay, then 200k gradient steps on the critics alone (8 per environment step: 25k steps) before the actor."""
    from f1sim import loop

    fresh = loop.schedule(False, rows=62_500, slots=64, updates=8, batch=256, n_step=5)
    assert fresh.learn_from == fresh.actor_from == fresh.full_from == 4 + 5 + 1
    assert float(fresh.actor_share(jnp.asarray(fresh.actor_from))) == 1.0  # from scratch: no ramp
    warm = loop.schedule(True, rows=62_500, slots=64, updates=8, batch=256, n_step=5, fill=0.2, critic_steps=200_000, actor_ramp=100_000)
    assert warm.learn_from == 12_500 and warm.actor_from == 12_500 + 25_000
    assert warm.full_from == warm.actor_from + 12_500  # 100k gradient steps at 8 per environment step
    share = lambda n: float(warm.actor_share(jnp.asarray(n)))
    assert share(warm.actor_from - 1) == 0.0 and 0.0 < share(warm.actor_from) < 0.001
    assert share(warm.actor_from + 6_250 - 1) == pytest.approx(0.5) and share(warm.full_from) == 1.0
    tiny = loop.schedule(True, rows=10, slots=64, updates=1, batch=256, n_step=5, fill=0.2, critic_steps=0)
    assert tiny.learn_from == tiny.actor_from == 10  # never before a batch can be drawn


def test_average_in_moves_the_averaged_actor_part_of_the_way():
    avg, now = {"w": jnp.zeros(3)}, {"w": jnp.full(3, 4.0)}
    out = stage_b.average_in(avg, now, 0.25)
    assert np.allclose(out["w"], 1.0) and np.allclose(avg["w"], 0.0)
    share = 1.0 - 0.5 ** (1000 / 100_000)  # a 100k-step half-life in 1000-step chunks
    for _ in range(100):
        avg = stage_b.average_in(avg, now, share)
    assert np.allclose(avg["w"], 2.0, atol=1e-3)  # halfway after one half-life


def test_warning_kinds_split_the_bits_and_the_log_shows_them():
    from f1env import racecontrol as rc

    from f1sim.drivers import env as de

    k = np.asarray(de.warning_kinds(jnp.array([0, rc.SPEEDING, rc.BLUE | rc.SLOW, rc.PASSING, rc.BLEND, rc.CLOSE | rc.FAR])))
    assert k.tolist() == [[0] * 7, [1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1]]
    r = {"black_flags_by": dict(zip(de.WARNING_KINDS, [3.0, 0.0, 1.0, 2.0, 0.0, 4.0, 1.0])), "warnings_by": dict(zip(de.WARNING_KINDS, [9.0, 1.0, 4.0, 5.0, 2.0, 7.0, 3.0]))}
    assert stage_b.kinds_text(r) == " [s3 p0 b1 m2 x0 c4 f1 | w s9 p1 b4 m5 x2 c7 f3]"
    assert stage_b.kinds_text({}) == ""


def test_a_best_only_replaces_its_own_personality():
    from f1sim.drivers.stage_b import oldest_slot

    owners, ages = ["middle"] * 4 + ["smooth"] * 4, list(range(8))
    seq = []
    for _ in range(6):
        k = oldest_slot(owners, ages, "smooth")
        seq.append(k)
        ages[k] = max(ages) + 1
    assert seq == [4, 5, 6, 7, 4, 5]  # the oldest smooth slot each time
    assert ages[:4] == [0, 1, 2, 3]  # the middle slots never change
    assert oldest_slot(owners, ages, "aggressive") is None  # no slot of its own: its bests don't go in


def test_trimming_the_heap_is_safe_to_call():
    from f1sim.drivers.stage_b import trim_heap

    trim_heap()  # glibc hands freed pages back; elsewhere a no-op


def test_frozen_cars_take_their_own_snapshots_action():
    enc = TINY["encoder"]
    actors = [tqc.init(jax.random.PRNGKey(k), tiny(1)).actor for k in range(3)]
    pool = stage_b.stack(actors)
    for races, frozen, k in ((4, 16, 3), (2, 7, 3), (3, 4, 3)):  # uneven shares too
        pool_k = stage_b.stack(actors[:k])
        pool_of = stage_b.assign_pool(jax.random.PRNGKey(races), races, frozen, k)
        obs = jax.random.uniform(jax.random.PRNGKey(9), (races, frozen, de.OBS))
        got = np.asarray(stage_b.pool_actions(pool_k, pool_of, obs, enc))
        want = np.stack([np.stack([np.asarray(nets.actor_mean(actors[int(pool_of[r, f])], enc, obs[r, f][None]))[0] for f in range(frozen)]) for r in range(races)])
        np.testing.assert_allclose(got, want, atol=1e-5)
